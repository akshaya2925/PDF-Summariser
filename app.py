import os
import io
import socket
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from PyPDF2 import PdfReader
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
from langdetect import detect
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM # Changed to Auto classes for flexibility
import torch
import threading

# Configuration and Initialization
# Set Tesseract path if it's not in your system's PATH.
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Remember to also install Poppler and add its bin directory to your PATH for pdf2image to work.

class PDFSummarizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PDF Summarizer (T5 Base Model)") # Updated title
        self.root.geometry("800x600")
        self.root.configure(bg='#f0f0f0')

        self.model = None
        self.tokenizer = None
        # Use "cuda" if a GPU is available, otherwise "cpu"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.file_path = None

        self.create_widgets()
        # Check internet and load model when the app starts
        self.check_internet_and_load_model()

    def create_widgets(self):
        """Initializes and places all Tkinter widgets in the application window."""
        frame = tk.Frame(self.root, bg='#f0f0f0', padx=20, pady=20)
        frame.pack(expand=True, fill='both')

        title = tk.Label(frame, text="PDF Summarizer", font=("Arial", 24, "bold"), bg='#f0f0f0', fg='#333')
        title.pack(pady=(0, 20))

        # File upload section
        file_frame = tk.Frame(frame, bg='#f0f0f0')
        file_frame.pack(fill='x', pady=(10, 5))
        
        self.file_label = tk.Label(file_frame, text="No file selected.", font=("Arial", 12), bg='#f0f0f0', fg='#555')
        self.file_label.pack(side='left', padx=(0, 10))

        self.browse_button = tk.Button(file_frame, text="Browse PDF", command=self.browse_file, font=("Arial", 10, "bold"), bg='#5a2e9b', fg='white', relief='flat', activebackground='#4a2a8b', activeforeground='white')
        self.browse_button.pack(side='left', padx=(0, 5))

        # Summarize button - initially disabled until model is loaded and file is selected
        self.summarize_button = tk.Button(file_frame, text="Summarize", command=self.start_summarization_thread, font=("Arial", 10, "bold"), bg='#2e9b5a', fg='white', relief='flat', state=tk.DISABLED, activebackground='#2a8b4a', activeforeground='white')
        self.summarize_button.pack(side='left', padx=(5, 0))

        # Progress and status messages
        self.status_label = tk.Label(frame, text="", font=("Arial", 10, "italic"), bg='#f0f0f0', fg='red')
        self.status_label.pack(pady=5)

        # Summary text area
        summary_label = tk.Label(frame, text="Summary:", font=("Arial", 12, "bold"), bg='#f0f0f0', fg='#333')
        summary_label.pack(pady=(15, 5), anchor='w')
        
        self.summary_text = scrolledtext.ScrolledText(frame, wrap=tk.WORD, font=("Arial", 11), height=20, borderwidth=1, relief="solid", bg='#ffffff', fg='#333')
        self.summary_text.pack(expand=True, fill='both')

    def check_internet_and_load_model(self):
        """Checks for internet connection and loads the summarization model in a separate thread."""
        self.status_label.config(text="Checking for internet connection...")
        if not self.is_connected():
            self.status_label.config(text="❌ No internet connection. Cannot load the summarization model.", fg='red')
            self.summarize_button.config(state=tk.DISABLED)
            return

        self.status_label.config(text="✅ Internet connected. Loading summarization model (t5-base)...", fg='blue') # Updated status
        # Start model loading in a background thread to prevent UI freeze
        threading.Thread(target=self.load_model).start()

    def is_connected(self):
        """
        Checks for an active internet connection by attempting to connect to Google.
        Returns True if connected, False otherwise.
        """
        try:
            # Try to connect to Google's public DNS server
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except OSError:
            return False

    def load_model(self):
        """
        Loads the Hugging Face T5-base model and tokenizer.
        Updates UI status on completion or error.
        """
        try:
            model_name = "t5-base" # CHANGED TO T5-BASE FOR BETTER QUALITY
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.model.to(self.device) # Move model to GPU if available
            # Update UI on the main thread after model loading
            self.root.after(0, lambda: self.status_label.config(text="✅ T5-base model loaded successfully!", fg='green')) # Updated status
            # Enable summarize button if a file is already selected
            self.root.after(0, lambda: self.summarize_button.config(state=tk.NORMAL if self.file_path else tk.DISABLED))
        except Exception as e:
            # Display error message if model loading fails
            self.root.after(0, lambda: self.status_label.config(text=f"❌ Failed to load model: {str(e)}. Please check your internet and try again.", fg='red'))

    def browse_file(self):
        """
        Opens a file dialog for the user to select a PDF file.
        Updates the file label and enables/disables the summarize button accordingly.
        """
        file_path = filedialog.askopenfilename(
            filetypes=[("PDF files", "*.pdf")]
        )
        if file_path:
            self.file_path = file_path
            self.file_label.config(text=os.path.basename(file_path))
            # Enable summarize button only if model is loaded
            self.summarize_button.config(state=tk.NORMAL if self.model else tk.DISABLED)
            self.status_label.config(text="") # Clear previous status messages
            self.summary_text.delete(1.0, tk.END) # Clear previous summary

    def start_summarization_thread(self):
        """
        Starts the summarization process in a new thread to keep the UI responsive.
        Disables the summarize button and updates status.
        """
        if not self.file_path:
            messagebox.showerror("Error", "Please select a PDF file first.")
            return

        self.summarize_button.config(state=tk.DISABLED) # Disable button immediately
        self.status_label.config(text="Processing PDF...", fg='blue')
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(tk.END, "Please wait, this may take a while, especially for long documents...\n")

        # Start summarization in a background thread
        threading.Thread(target=self.process_summarization).start()

    def process_summarization(self):
        """
        Contains the main text extraction, language detection, and summarization logic.
        Updates the UI with the summary or error messages.
        """
        try:
            # Attempt to get text from PDF (PyPDF2 first, then OCR if needed)
            text = self.get_text_from_pdf(self.file_path)

            if not text.strip():
                self.root.after(0, lambda: messagebox.showerror("Error", "No readable text found in the PDF. It might be an image-only PDF with unclear text, or badly formatted."))
                self.root.after(0, lambda: self.summarize_button.config(state=tk.NORMAL))
                return

            # Language detection
            try:
                language = detect(text)
                if language != 'en':
                    self.root.after(0, lambda: messagebox.showerror("Error", f"I can only summarize English text. The detected language is '{language}'."))
                    self.root.after(0, lambda: self.summarize_button.config(state=tk.NORMAL))
                    return
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Could not detect language. The text might be too short or unusual: {str(e)}"))
                self.root.after(0, lambda: self.summarize_button.config(state=tk.NORMAL))
                return

            # Perform summarization
            summary = self.summarize_text(text)
            
            # Update UI with the summary and status
            self.root.after(0, lambda: self.summary_text.delete(1.0, tk.END))
            self.root.after(0, lambda: self.summary_text.insert(tk.END, summary))
            self.root.after(0, lambda: self.status_label.config(text="✅ Summarization complete!", fg='green'))
            self.root.after(0, lambda: self.summarize_button.config(state=tk.NORMAL))

        except Exception as e:
            # Catch any unexpected errors during the process
            self.root.after(0, lambda: messagebox.showerror("Error", f"An unexpected error occurred: {str(e)}"))
            self.root.after(0, lambda: self.summarize_button.config(state=tk.NORMAL))
            
    def get_text_from_pdf(self, pdf_path):
        """
        Extracts text from a PDF. Attempts direct text extraction first.
        If no text is found, falls back to OCR.
        """
        text = self.extract_text_from_pdf(pdf_path)
        if not text.strip():
            self.root.after(0, lambda: self.status_label.config(text="No direct text found. Attempting OCR...", fg='orange'))
            text = self.ocr_from_pdf(pdf_path)
        return text

    def extract_text_from_pdf(self, pdf_path):
        """
        Extracts text from a digitally native PDF using PyPDF2.
        Returns extracted text or an empty string on failure.
        """
        try:
            reader = PdfReader(pdf_path)
            full_text = ""
            for i, page in enumerate(reader.pages):
                self.root.after(0, lambda i=i: self.status_label.config(text=f"Extracting text from page {i+1}/{len(reader.pages)}...", fg='blue'))
                full_text += page.extract_text() or ""
            return full_text
        except Exception as e:
            self.root.after(0, lambda: self.status_label.config(text=f"Error extracting text with PyPDF2: {str(e)}", fg='red'))
            return ""

    def ocr_from_pdf(self, pdf_path):
        """
        Performs Optical Character Recognition (OCR) on a scanned PDF.
        Requires Tesseract and Poppler to be installed and accessible.
        Returns OCR'd text or an empty string on failure.
        """
        try:
            # Convert PDF pages to images
            self.root.after(0, lambda: self.status_label.config(text="Converting PDF to images for OCR...", fg='blue'))
            # If Poppler is not in PATH, specify poppler_path here:
            # images = convert_from_path(pdf_path, poppler_path=r'C:\path\to\poppler\bin')
            images = convert_from_path(pdf_path) 
            full_text = ""
            for i, image in enumerate(images):
                self.root.after(0, lambda i=i: self.status_label.config(text=f"OCR'ing page {i+1}/{len(images)}...", fg='blue'))
                full_text += pytesseract.image_to_string(image)
            return full_text
        except pytesseract.TesseractNotFoundError:
            self.root.after(0, lambda: messagebox.showerror("OCR Error", "Tesseract OCR engine not found. Please install it and set the path in the code if necessary."))
            self.root.after(0, lambda: self.status_label.config(text="❌ Tesseract not found.", fg='red'))
            return ""
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("OCR Error", f"An error occurred during OCR: {str(e)}. Make sure Poppler is installed and in your PATH."))
            self.root.after(0, lambda: self.status_label.config(text=f"❌ OCR failed: {str(e)}", fg='red'))
            return ""

    def summarize_text(self, text):
        """
        Summarizes text using the loaded T5 model. Handles long documents by chunking
        and iteratively summarizing.
        """
        # T5 models expect the input prefixed with "summarize: "
        # T5-base has a max input length of 512 tokens.
        max_chunk_size = 512 

        # Tokenize the entire text
        # The 'truncation=False' ensures we get all tokens before chunking manually
        tokens = self.tokenizer.encode(text, return_tensors='pt', truncation=False)
        # Move tokens to the correct device (CPU/GPU)
        tokens = tokens.to(self.device)

        # If the text is short enough, summarize directly
        if tokens.shape[1] <= max_chunk_size:
            self.root.after(0, lambda: self.status_label.config(text="Summarizing document...", fg='blue'))
            inputs = self.tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=max_chunk_size, truncation=True).to(self.device)
            summary_ids = self.model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
            return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # For longer texts, split into overlapping chunks to maintain context
        # A smaller chunk_size for actual processing to allow for overlap and prefix
        processing_chunk_size = max_chunk_size - 50 # Leave some buffer for "summarize: " prefix and overlap
        overlap_size = 50 # Overlap between chunks to maintain context
        chunks = []
        for i in range(0, tokens.shape[1], processing_chunk_size - overlap_size):
            chunk = tokens[:, i:i+processing_chunk_size]
            chunks.append(chunk)

        summaries = []
        for i, chunk in enumerate(chunks):
            self.root.after(0, lambda i=i: self.status_label.config(text=f"Summarizing chunk {i+1}/{len(chunks)}...", fg='blue'))
            # Decode the chunk to text, add the prefix, and re-tokenize for the model
            chunk_text = self.tokenizer.decode(chunk[0], skip_special_tokens=True)
            input_ids = self.tokenizer.encode("summarize: " + chunk_text, return_tensors="pt", max_length=max_chunk_size, truncation=True).to(self.device)
            
            summary_ids = self.model.generate(input_ids, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
            summaries.append(self.tokenizer.decode(summary_ids[0], skip_special_tokens=True))
        
        combined_summary = " ".join(summaries)
        
        # If the combined summary is still very long, re-summarize it for conciseness
        # The threshold (e.g., 500 words) can be adjusted based on desired output length
        if len(combined_summary.split()) > 500:
            self.root.after(0, lambda: self.status_label.config(text="Finalizing summary by re-summarizing combined chunks...", fg='blue'))
            return self.summarize_text(combined_summary) # Recursively call to re-summarize

        return combined_summary

if __name__ == "__main__":
    root = tk.Tk()
    app = PDFSummarizerApp(root)
    root.mainloop()
