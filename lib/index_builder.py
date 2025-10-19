from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.datamodel.base_models import InputFormat
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document as LangchainDocument
from lib.hybrid_retriever import EnhancedHybridRetriever
import os
import re
import shutil
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logging.getLogger('RapidOCR').setLevel(logging.ERROR)
logging.getLogger('docling').setLevel(logging.WARNING)

# Title mapping
TITLE_MAPPING = {
    "Deep_Learning_Method_to_Detect_the_Road_Cracks_and.pdf": "Deep Learning Method to Detect the Road Cracks and Potholes for Smart Cities",
    "Detection_of_Potholes_on_Roads_using_a_Drone.pdf": "Detection of Potholes on Roads using a Drone",
    "1-s2.0-S2095756423000272-main.pdf": "Automated classification and detection of multiple pavement distress images based on deep learning",
    "22_potholedetectionusingdeeplearning107-115247AE.pdf": "Road Pothole Detection Using Unmanned Aerial Vehicle Imagery and Deep Learning Technique",
    "document1.pdf": "An Improved Mobile Sensing Algorithm for Potholes Detection and Analysis",
    "IJARCCE99.pdf": "IJARCCE RIPD: Route Information and Pothole Detection",
    "Image-Based_Pothole_Detection_System_for_ITS_Servi.pdf": "Image-Based Pothole Detection System for ITS Service and Road Management System",
    "ITCN-05.pdf": "Pothole Detection with Image Processing and Spectral Clustering",
    "Method_for_Automated_Assessment_of_Potholes_Cracks.pdf": "Method for Automated Assessment of Potholes, Cracks and Patches from Road Surface Video Clips",
    "Pothole_Detection_An_Efficient_Vision_Based_Method.pdf": "Pothole Detection: An Efficient Vision Based Method Using RGB Color Space Image Segmentation",
    "Pothole_detection_and_dimension_estimation_by_deep.pdf": "Pothole Detection and Dimension Estimation by Deep Learning",
    "PotholePublishedPaper.pdf": "Convolutional Neural Networks Based Potholes Detection Using Thermal Imaging",
    "Road_Surface_Automatic_Identification_System_With_.pdf": "Road Surface Automatic Identification System With Combination Pothole Detection Method and Z-Diff Method On Android Smartphone",
    "sustainability-16-09168-v2 (most relevant paper).pdf": "PDS-UAV: A Deep Learning-Based Pothole Detection System Using Unmanned Aerial Vehicle Images"
}

def create_enhanced_converter():
    """
    Create DocumentConverter with OCR and enhanced table extraction
    """
    pipeline_options = PdfPipelineOptions()
    
    # Enable OCR for images and scanned content
    pipeline_options.do_ocr = True
    
    # Enhanced table extraction
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
    
    # Image processing
    pipeline_options.images_scale = 2.0
    pipeline_options.generate_picture_images = True
    
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )
    
    logger.info("Enhanced DocumentConverter created with OCR enabled")
    return converter

def extract_visual_content_metadata(result, title):
    """
    Extract metadata from tables, figures, and visual content
    ChromaDB Compatible - No lists, only simple types
    """
    metadata = {}
    
    try:
        # Extract tables
        if hasattr(result.document, 'tables') and result.document.tables:
            table_count = len(result.document.tables)
            metadata['table_count'] = int(table_count)
            logger.info(f"  Found {table_count} tables in {title}")
            
            # Extract accuracy from tables
            accuracies = []
            for table in result.document.tables:
                table_text = str(table).lower()
                accuracy_patterns = [
                    r'(\d+\.?\d*)\s*%',
                    r'accuracy[:\s]+(\d+\.?\d*)',
                    r'precision[:\s]+(\d+\.?\d*)',
                ]
                for pattern in accuracy_patterns:
                    matches = re.findall(pattern, table_text)
                    for match in matches:
                        try:
                            acc = float(match)
                            if 0 < acc <= 100:
                                accuracies.append(acc)
                        except:
                            pass
            
            if accuracies:
                # FIXED: Store as comma-separated string, not list
                metadata['table_accuracies_str'] = ", ".join(f"{a:.2f}" for a in accuracies)
                metadata['max_table_accuracy'] = float(max(accuracies))
                metadata['table_accuracy_count'] = int(len(accuracies))
                logger.info(f"  Extracted accuracies from tables: {accuracies}")
        
        # Extract figures/images
        if hasattr(result.document, 'pictures') and result.document.pictures:
            figure_count = len(result.document.pictures)
            metadata['figure_count'] = int(figure_count)
            logger.info(f"  Found {figure_count} figures/images in {title}")
            
            # Extract captions
            captions = []
            for pic in result.document.pictures:
                if hasattr(pic, 'caption') and pic.caption:
                    captions.append(str(pic.caption))
            
            if captions:
                # Store as single string with separator
                metadata['figure_captions_str'] = " | ".join(captions[:5])
                metadata['caption_count'] = int(len(captions))
        
    except Exception as e:
        logger.warning(f"Error extracting visual content metadata: {e}")
    
    return metadata

def get_title_from_mapping(pdf_path):
    """Get title from manual mapping"""
    filename = os.path.basename(pdf_path)
    
    if filename in TITLE_MAPPING:
        title = TITLE_MAPPING[filename]
        logger.info(f"Using mapped title for {filename}: {title}")
        return title
    else:
        title = filename.replace('.pdf', '').replace('_', ' ').replace('-', ' ')
        title = re.sub(r'\s+', ' ', title).strip()
        title = ' '.join(word.capitalize() for word in title.split())
        logger.info(f"Using fallback title for {filename}: {title}")
        return title

def extract_metadata_from_content(content, title):
    """
    Extract enhanced metadata from content
    ChromaDB Compatible - No lists, only str/int/float/bool
    """
    metadata = {}
    content_lower = content.lower()
    
    # Topic extraction
    topics = []
    if any(keyword in content_lower for keyword in ['pothole', 'potholes']):
        topics.append("pothole detection")
    if any(keyword in content_lower for keyword in ['crack', 'cracks', 'road crack']):
        topics.append("road cracks")
    if any(keyword in content_lower for keyword in ['pavement', 'asphalt']):
        topics.append("pavement analysis")
    if any(keyword in content_lower for keyword in ['drone', 'uav', 'aerial']):
        topics.append("drone-based detection")
    if any(keyword in content_lower for keyword in ['thermal', 'infrared']):
        topics.append("thermal imaging")
    if any(keyword in content_lower for keyword in ['mobile', 'smartphone', 'android']):
        topics.append("mobile detection")
    
    if topics:
        metadata["topics"] = ", ".join(topics)
    
    # Method detection
    methods = []
    if 'cnn' in content_lower or 'convolutional neural network' in content_lower:
        methods.append("CNN")
    if 'yolo' in content_lower:
        methods.append("YOLO")
    if 'rcnn' in content_lower or 'r-cnn' in content_lower:
        methods.append("R-CNN")
    if 'svm' in content_lower or 'support vector machine' in content_lower:
        methods.append("SVM")
    if 'deep learning' in content_lower:
        methods.append("Deep Learning")
    if 'machine learning' in content_lower:
        methods.append("Machine Learning")
    
    if methods:
        metadata["methods"] = ", ".join(methods)
    
    # Enhanced accuracy extraction
    accuracy_patterns = [
        r'(\d+\.?\d*)\s*%\s*(?:accuracy|precision|recall|f1)',
        r'(?:accuracy|precision|recall|f1)[:\s]+(\d+\.?\d*)\s*%?',
        r'achieved\s+(?:an?\s+)?accuracy\s+of\s+(\d+\.?\d*)',
        r'accuracy\s+(?:of|was|is)\s+(\d+\.?\d*)',
    ]
    
    all_accuracies = []
    for pattern in accuracy_patterns:
        matches = re.findall(pattern, content_lower[:10000])
        for match in matches:
            try:
                acc = float(match)
                if 0 < acc <= 100:
                    all_accuracies.append(acc)
            except:
                pass
    
    # FIXED: ChromaDB compatible - no lists
    if all_accuracies:
        # Store as comma-separated string
        metadata["accuracies_found_str"] = ", ".join(f"{a:.2f}" for a in all_accuracies)
        metadata["accuracy_count"] = int(len(all_accuracies))
        
        # Store individual numeric values
        metadata["max_accuracy_mentioned"] = float(max(all_accuracies))
        metadata["min_accuracy_mentioned"] = float(min(all_accuracies))
        metadata["avg_accuracy_mentioned"] = float(sum(all_accuracies) / len(all_accuracies))
    
    return metadata

def index_loader(force_rebuild=False):
    """Load or rebuild the vector index with enhanced OCR"""
    persist_dir = "chroma_db"
    embd = OpenAIEmbeddings()

    if force_rebuild and os.path.exists(persist_dir):
        logger.info("Force rebuilding Chroma database...")
        shutil.rmtree(persist_dir)

    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        logger.info("Loading existing Chroma database...")
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embd,
            collection_name="rag-chroma-pdfs",
        )
        
        # Reconstruct documents for BM25
        all_docs = vectorstore.get()
        documents = []
        
        if all_docs and 'ids' in all_docs and len(all_docs['ids']) > 0:
            for i in range(len(all_docs['ids'])):
                doc = LangchainDocument(
                    page_content=all_docs['documents'][i],
                    metadata=all_docs['metadatas'][i] if all_docs['metadatas'] else {}
                )
                documents.append(doc)
            
            logger.info(f"Loaded {len(documents)} documents for hybrid retrieval")
            return EnhancedHybridRetriever(vectorstore, documents)
        else:
            logger.warning("Database exists but is empty. Rebuilding...")
            shutil.rmtree(persist_dir)

    logger.info("=" * 70)
    logger.info("Creating database with ENHANCED OCR and visual content")
    logger.info("=" * 70)

    pdf_directory = "research_papers"
    paths = []
    if os.path.exists(pdf_directory):
        for filename in os.listdir(pdf_directory):
            if filename == "CCECE-2016.pdf":
                logger.info(f"SKIPPING {filename} - excluded")
                continue
                
            if filename.lower().endswith('.pdf'):
                paths.append(os.path.join(pdf_directory, filename))
    
    if not paths:
        raise Exception(f"No PDF files found in {pdf_directory}")

    logger.info(f"Found {len(paths)} PDF files to process with OCR")
    
    converter = create_enhanced_converter()
    
    docs = []
    processed_files = []
    failed_files = []
    total_tables = 0
    total_figures = 0
    
    for pdf_path in paths:
        if not os.path.exists(pdf_path):
            logger.warning(f"File not found: {pdf_path}")
            failed_files.append(os.path.basename(pdf_path))
            continue
            
        filename = os.path.basename(pdf_path)
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing: {filename}")
        logger.info(f"{'='*70}")
        
        try:
            result = converter.convert(pdf_path)
            markdown_content = result.document.export_to_markdown()
            
            if len(markdown_content.strip()) < 300:
                logger.info(f"Skipping short content: {filename}")
                failed_files.append(filename)
                continue
            
            title = get_title_from_mapping(pdf_path)
            
            # Extract visual content metadata (ChromaDB compatible)
            visual_metadata = extract_visual_content_metadata(result, title)
            if visual_metadata.get('table_count'):
                total_tables += visual_metadata['table_count']
            if visual_metadata.get('figure_count'):
                total_figures += visual_metadata['figure_count']
            
            enhanced_content = f"""RESEARCH PAPER TITLE: {title}

DOCUMENT CONTENT (INCLUDING TABLES, FIGURES, AND OCR TEXT):
{markdown_content}

---
[Paper: {title}]
"""
            
            metadata = {
                "source": pdf_path,
                "title": title,
                "filename": filename,
                "document_type": "research_paper",
            }
            
            # Merge text and visual metadata (all ChromaDB compatible)
            text_metadata = extract_metadata_from_content(markdown_content, title)
            metadata.update(text_metadata)
            metadata.update(visual_metadata)
            
            doc = LangchainDocument(
                page_content=enhanced_content,
                metadata=metadata
            )
            docs.append(doc)
            processed_files.append(filename)
            
            logger.info(f"✅ Processed: {title}")
            if metadata:
                logger.info(f"  Metadata extracted:")
                for key, value in metadata.items():
                    if key not in ['source', 'filename', 'document_type', 'title']:
                        logger.info(f"    - {key}: {value}")
            
        except Exception as e:
            logger.error(f"❌ Error processing {pdf_path}: {e}")
            failed_files.append(filename)
            continue

    if not docs:
        raise Exception("No documents processed successfully")

    logger.info(f"\n{'='*70}")
    logger.info(f"PROCESSING SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"Successfully processed: {len(docs)}/{len(paths)} PDFs")
    logger.info(f"Total tables extracted: {total_tables}")
    logger.info(f"Total figures extracted: {total_figures}")
    if failed_files:
        logger.warning(f"Failed files: {failed_files}")

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1200,
        chunk_overlap=200,
    )
    
    doc_chunks = text_splitter.split_documents(docs)
    logger.info(f"Split into {len(doc_chunks)} chunks")

    # Create vectorstore with ChromaDB-compatible metadata
    vectorstore = Chroma.from_documents(
        documents=doc_chunks,
        collection_name="rag-chroma-pdfs",
        embedding=embd,
        persist_directory=persist_dir,
    )
    
    logger.info("✅ Database created with OCR and visual content extraction")
    
    return EnhancedHybridRetriever(vectorstore, doc_chunks)

def get_title_mapping():
    """Get the manual title mapping"""
    return TITLE_MAPPING.copy()

def get_expected_titles_count():
    """Get the number of expected titles"""
    return len(TITLE_MAPPING)

def get_all_titles():
    """Get all titles from the manual mapping"""
    return list(TITLE_MAPPING.values())

def get_titles_with_filenames():
    """Get titles with their corresponding filenames"""
    return TITLE_MAPPING.copy()