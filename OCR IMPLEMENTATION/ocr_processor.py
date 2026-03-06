from __future__ import annotations

import logging
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol

import fitz  # PyMuPDF

logger = logging.getLogger("sourcesleuth.ocr_processor")

DEFAULT_OCR_LANG = "eng"
DEFAULT_OCR_DPI = 300
DEFAULT_OCR_DPI_HIGH = 600
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
APPROX_CHARS_PER_TOKEN = 4
DEFAULT_APPLY_POSTPROCESSING = True


@dataclass
class TextChunk:
    text: str
    filename: str
    page: int
    chunk_index: int
    start_char: int
    end_char: int
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "filename": self.filename,
            "page": self.page,
            "chunk_index": self.chunk_index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TextChunk":
        return cls(**data)


@dataclass
class PageSpan:
    page: int
    start_char: int
    end_char: int
    confidence: float = 1.0


@dataclass
class OCRDocument:
    filename: str
    full_text: str
    page_spans: list[PageSpan] = field(default_factory=list)
    chunks: list[TextChunk] = field(default_factory=list)
    ocr_engine: str = ""


class OCRStrategy(Protocol):
    def extract_text(self, image_path: Path) -> tuple[str, float]:
        ...


class ImagePreprocessor(ABC):
    @abstractmethod
    def preprocess(self, image):
        pass


class DefaultImagePreprocessor(ImagePreprocessor):
    def __init__(self, contrast_factor: float = 1.3, blur_radius: float = 0.5):
        self.contrast_factor = contrast_factor
        self.blur_radius = blur_radius

    def preprocess(self, image):
        from PIL import Image, ImageFilter, ImageEnhance

        if image.mode != 'L':
            image = image.convert('L')

        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(self.contrast_factor)
        image = image.filter(ImageFilter.SHARPEN)
        image = image.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))

        return image


class EasyOCREngine:
    def __init__(self):
        self._reader = None

    def _get_reader(self):
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        return self._reader

    def extract(self, image_path: Path, paragraph: bool = True) -> tuple[str, float]:
        reader = self._get_reader()
        results = reader.readtext(str(image_path), min_size=10, text_threshold=0.7, paragraph=paragraph)

        if not results:
            return "", 0.0

        texts = []
        confidences = []

        for result in results:
            if len(result) == 3:
                _, text, confidence = result
                confidences.append(confidence)
            elif len(result) == 2:
                _, text = result
                confidences.append(1.0)
            else:
                continue
            texts.append(text)

        full_text = " ".join(texts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        if DEFAULT_APPLY_POSTPROCESSING:
            try:
                from .ocr_postprocessor import postprocess_ocr_text
                full_text, postproc_stats = postprocess_ocr_text(full_text, avg_confidence)
                if postproc_stats.get('corrections_applied', 0) > 0:
                    logger.info(
                        "Post-processing applied %d corrections to EasyOCR output from '%s'",
                        postproc_stats['corrections_applied'], image_path.name
                    )
            except ImportError:
                logger.debug("Post-processor not available, using raw OCR output")

        logger.info(
            "EasyOCR extracted %d characters from '%s' (confidence: %.2f)",
            len(full_text), image_path.name, avg_confidence
        )

        return full_text, avg_confidence


class TesseractEngine:
    def __init__(self, preprocessor: Optional[ImagePreprocessor] = None):
        self.preprocessor = preprocessor or DefaultImagePreprocessor()

    def extract(self, image_path: Path, enhance: bool = True) -> tuple[str, float]:
        import pytesseract
        from PIL import Image

        logger.info("Using pytesseract for image: %s", image_path.name)

        image = Image.open(image_path)
        if enhance:
            image = self.preprocessor.preprocess(image)

        custom_config = r'--oem 3 --psm 6'
        ocr_data = pytesseract.image_to_data(
            image, output_type=pytesseract.Output.DICT, config=custom_config
        )

        texts = []
        confidences = []

        for i, text in enumerate(ocr_data['text']):
            if text.strip():
                texts.append(text)
                conf = int(ocr_data['conf'][i])
                if conf > 0:
                    confidences.append(conf / 100.0)

        full_text = " ".join(texts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        logger.info(
            "pytesseract extracted %d characters from '%s' (confidence: %.2f)",
            len(full_text), image_path.name, avg_confidence
        )

        return full_text, avg_confidence


class OCRFactory:
    @staticmethod
    def get_easyocr() -> EasyOCREngine:
        return EasyOCREngine()

    @staticmethod
    def get_tesseract(preprocessor: Optional[ImagePreprocessor] = None) -> TesseractEngine:
        return TesseractEngine(preprocessor)


def check_tesseract_availability() -> bool:
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def check_easyocr_availability() -> bool:
    try:
        import easyocr
        return True
    except ImportError:
        return False


class TextExtractor:
    def __init__(
        self,
        easyocr_engine: Optional[EasyOCREngine] = None,
        tesseract_engine: Optional[TesseractEngine] = None,
    ):
        self.easyocr_engine = easyocr_engine
        self.tesseract_engine = tesseract_engine

    def extract_from_image(self, image_path: Path, enhance: bool = True) -> tuple[str, float]:
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        if self.easyocr_engine:
            return self.easyocr_engine.extract(image_path)

        if self.tesseract_engine:
            return self.tesseract_engine.extract(image_path, enhance=enhance)

        raise RuntimeError(
            "No OCR engine available. Install 'easyocr' or 'pytesseract' with Tesseract-OCR."
        )


def create_text_extractor() -> TextExtractor:
    easyocr_available = check_easyocr_availability()
    tesseract_available = check_tesseract_availability()

    easyocr_engine = OCRFactory.get_easyocr() if easyocr_available else None
    tesseract_engine = OCRFactory.get_tesseract() if tesseract_available else None

    return TextExtractor(easyocr_engine, tesseract_engine)


def extract_text_from_image_basic(image_path: str | Path, enhance: bool = True) -> tuple[str, float]:
    extractor = create_text_extractor()
    return extractor.extract_from_image(Path(image_path), enhance=enhance)


def extract_text_from_image_pdf(
    pdf_path: str | Path,
    dpi: int = DEFAULT_OCR_DPI,
    ocr_lang: str = DEFAULT_OCR_LANG,
    enhance: bool = True,
) -> OCRDocument:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        raise RuntimeError(f"Failed to open PDF '{pdf_path.name}': {exc}") from exc

    extractor = create_text_extractor()
    full_text_parts: list[str] = []
    page_spans: list[PageSpan] = []
    current_offset = 0
    ocr_engine_used = ""

    for page_num in range(len(doc)):
        page = doc[page_num]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(img_data)
            tmp_path = Path(tmp.name)

        try:
            page_text, confidence = extractor.extract_from_image(tmp_path)
            ocr_engine_used = "easyocr" if check_easyocr_availability() else "pytesseract"
        finally:
            tmp_path.unlink()

        if not page_text.strip():
            logger.warning("No text extracted from page %d of '%s'", page_num + 1, pdf_path.name)
            continue

        start = current_offset
        full_text_parts.append(page_text)
        current_offset += len(page_text)

        page_spans.append(PageSpan(
            page=page_num + 1,
            start_char=start,
            end_char=current_offset,
            confidence=confidence,
        ))

    doc.close()

    full_text = "".join(full_text_parts)
    logger.info(
        "OCR extracted %d characters from %d pages of '%s' (engine: %s)",
        len(full_text), len(page_spans), pdf_path.name, ocr_engine_used,
    )

    return OCRDocument(
        filename=pdf_path.name,
        full_text=full_text,
        page_spans=page_spans,
        ocr_engine=ocr_engine_used,
    )


def extract_text_from_image_file(image_path: str | Path) -> tuple[str, float]:
    return extract_text_from_image_basic(image_path)


def _char_size(token_count: int) -> int:
    return token_count * APPROX_CHARS_PER_TOKEN


class TextChunker:
    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: OCRDocument) -> list[TextChunk]:
        text = document.full_text
        if not text.strip():
            logger.warning("OCR document '%s' has no extractable text.", document.filename)
            return []

        char_chunk = _char_size(self.chunk_size)
        char_overlap = _char_size(self.chunk_overlap)
        stride = max(char_chunk - char_overlap, 1)

        chunks: list[TextChunk] = []
        idx = 0
        start = 0

        while start < len(text):
            end = min(start + char_chunk, len(text))
            chunk_text_str = text[start:end].strip()

            if chunk_text_str:
                page, confidence = self._resolve_page(document.page_spans, start)
                chunks.append(TextChunk(
                    text=chunk_text_str,
                    filename=document.filename,
                    page=page,
                    chunk_index=idx,
                    start_char=start,
                    end_char=end,
                    confidence=confidence,
                ))
                idx += 1

            start += stride

        document.chunks = chunks
        logger.info(
            "Chunked OCR document '%s' into %d chunks (size=%d, overlap=%d tokens).",
            document.filename, len(chunks), self.chunk_size, self.chunk_overlap,
        )
        return chunks

    def _resolve_page(self, page_spans: list[PageSpan], char_offset: int) -> tuple[int, float]:
        for span in page_spans:
            if span.start_char <= char_offset < span.end_char:
                return span.page, span.confidence
        if page_spans:
            return page_spans[-1].page, page_spans[-1].confidence
        return 1, 1.0


def chunk_text(
    document: OCRDocument,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[TextChunk]:
    chunker = TextChunker(chunk_size, chunk_overlap)
    return chunker.chunk(document)


class PDFTextDetector:
    def __init__(self, min_chars_threshold: int = 100, avg_chars_threshold: float = 50):
        self.min_chars_threshold = min_chars_threshold
        self.avg_chars_threshold = avg_chars_threshold

    def needs_ocr(self, pdf_path: Path) -> bool:
        if not pdf_path.exists():
            return False

        try:
            doc = fitz.open(str(pdf_path))
        except Exception:
            return False

        total_text_length = 0
        page_count = len(doc)

        for page_num in range(page_count):
            page = doc[page_num]
            page_text = page.get_text("text")
            total_text_length += len(page_text.strip())

        doc.close()

        avg_chars_per_page = total_text_length / max(page_count, 1)
        needs_ocr = total_text_length < self.min_chars_threshold or avg_chars_per_page < self.avg_chars_threshold

        if needs_ocr:
            logger.info(
                "PDF '%s' appears to be image-only (%d chars, %.1f avg/page). OCR recommended.",
                pdf_path.name, total_text_length, avg_chars_per_page
            )
        else:
            logger.info(
                "PDF '%s' has sufficient text (%d chars). Standard extraction recommended.",
                pdf_path.name, total_text_length
            )

        return needs_ocr


def should_use_ocr(pdf_path: str | Path) -> bool:
    detector = PDFTextDetector()
    return detector.needs_ocr(Path(pdf_path))


class PDFProcessor(ABC):
    @abstractmethod
    def process(self, pdf_path: Path) -> list[TextChunk]:
        pass


class OCRPDFProcessor(PDFProcessor):
    def __init__(
        self,
        chunker: TextChunker,
        dpi: int = DEFAULT_OCR_DPI,
        enhance: bool = True,
    ):
        self.chunker = chunker
        self.dpi = dpi
        self.enhance = enhance

    def process(self, pdf_path: Path) -> list[TextChunk]:
        document = extract_text_from_image_pdf(pdf_path, dpi=self.dpi, enhance=self.enhance)
        return self.chunker.chunk(document)


class StandardPDFProcessor(PDFProcessor):
    def __init__(self, chunker: TextChunker):
        self.chunker = chunker

    def process(self, pdf_path: Path) -> list[TextChunk]:
        from .pdf_processor import extract_text_from_pdf

        document_std = extract_text_from_pdf(pdf_path)
        document = OCRDocument(
            filename=document_std.filename,
            full_text=document_std.full_text,
            page_spans=[
                PageSpan(page=ps.page, start_char=ps.start_char,
                        end_char=ps.end_char, confidence=1.0)
                for ps in document_std.page_spans
            ],
            ocr_engine="none",
        )
        return self.chunker.chunk(document)


class SmartPDFProcessor:
    def __init__(
        self,
        ocr_processor: PDFProcessor,
        standard_processor: PDFProcessor,
        detector: PDFTextDetector,
        force_ocr: bool = False,
    ):
        self.ocr_processor = ocr_processor
        self.standard_processor = standard_processor
        self.detector = detector
        self.force_ocr = force_ocr

    def process(self, pdf_path: Path) -> list[TextChunk]:
        use_ocr = self.force_ocr or self.detector.needs_ocr(pdf_path)

        if use_ocr:
            logger.info("Using OCR for '%s'", pdf_path.name)
            return self.ocr_processor.process(pdf_path)
        else:
            logger.info("Using standard extraction for '%s'", pdf_path.name)
            return self.standard_processor.process(pdf_path)


class DirectoryProcessor:
    def __init__(self, pdf_processor: PDFProcessor):
        self.pdf_processor = pdf_processor

    def process_directory(self, directory: Path) -> list[TextChunk]:
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a valid directory: {directory}")

        pdf_files = sorted(directory.glob("*.pdf"))
        if not pdf_files:
            logger.warning("No PDF files found in '%s'.", directory)
            return []

        all_chunks: list[TextChunk] = []

        for pdf_path in pdf_files:
            try:
                chunks = self.pdf_processor.process(pdf_path)
                all_chunks.extend(chunks)
                logger.info(
                    "Processed '%s' -> %d chunks",
                    pdf_path.name, len(chunks)
                )
            except Exception as exc:
                logger.error("Failed to process '%s': %s", pdf_path.name, exc)

        logger.info(
            "Total: processed %d PDFs -> %d chunks from '%s'.",
            len(pdf_files), len(all_chunks), directory,
        )
        return all_chunks


def process_ocr_pdf_directory(
    directory: str | Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    dpi: int = DEFAULT_OCR_DPI,
    force_ocr: bool = False,
    enhance: bool = True,
) -> list[TextChunk]:
    directory = Path(directory)

    chunker = TextChunker(chunk_size, chunk_overlap)
    ocr_processor = OCRPDFProcessor(chunker, dpi=dpi, enhance=enhance)
    standard_processor = StandardPDFProcessor(chunker)
    detector = PDFTextDetector()
    smart_processor = SmartPDFProcessor(
        ocr_processor, standard_processor, detector, force_ocr
    )
    processor = DirectoryProcessor(smart_processor)

    return processor.process_directory(directory)


def process_image_directory(
    directory: str | Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    image_extensions: list[str] | None = None,
) -> list[TextChunk]:
    if image_extensions is None:
        image_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"]

    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a valid directory: {directory}")

    chunker = TextChunker(chunk_size, chunk_overlap)
    extractor = create_text_extractor()
    all_chunks: list[TextChunk] = []

    for ext in image_extensions:
        image_files = sorted(directory.glob(f"*{ext}"))

        for image_path in image_files:
            try:
                text, confidence = extractor.extract_from_image(image_path)

                if not text.strip():
                    logger.warning("No text extracted from '%s'", image_path.name)
                    continue

                document = OCRDocument(
                    filename=image_path.name,
                    full_text=text,
                    page_spans=[PageSpan(page=1, start_char=0,
                                         end_char=len(text), confidence=confidence)],
                    ocr_engine="easyocr" if check_easyocr_availability() else "pytesseract",
                )

                chunks = chunker.chunk(document)
                all_chunks.extend(chunks)
                logger.info(
                    "Processed '%s' -> %d chunks (confidence: %.2f)",
                    image_path.name, len(chunks), confidence
                )
            except Exception as exc:
                logger.error("Failed to process '%s': %s", image_path.name, exc)

    logger.info(
        "Total: processed images in '%s' -> %d chunks.",
        directory, len(all_chunks),
    )
    return all_chunks
