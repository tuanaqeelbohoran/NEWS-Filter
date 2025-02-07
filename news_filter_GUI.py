import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
import pandas as pd
from sentence_transformers import SentenceTransformer, util

class NewsFilterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("News Filtering Tool")
        self.root.geometry("800x600")

        # Initialize the SentenceTransformer model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Create widgets
        self.create_widgets()

    def create_widgets(self):
        # File selection
        self.file_label = tk.Label(self.root, text="Select JSONL File:")
        self.file_label.pack(pady=10)
        self.file_button = tk.Button(self.root, text="Browse", command=self.select_file)
        self.file_button.pack(pady=5)

        # Query input
        self.query_label = tk.Label(self.root, text="Enter Query:")
        self.query_label.pack(pady=10)
        self.query_entry = tk.Entry(self.root, width=50)
        self.query_entry.pack(pady=5)

        # Filter button
        self.filter_button = tk.Button(self.root, text="Filter Articles", command=self.filter_articles)
        self.filter_button.pack(pady=20)

        # Result display
        self.result_text = ScrolledText(self.root, width=90, height=20)
        self.result_text.pack(pady=10)

        # Save button
        self.save_button = tk.Button(self.root, text="Save Results", command=self.save_results)
        self.save_button.pack(pady=10)

        # Initial state
        self.file_path = None
        self.filtered_articles = None

    def select_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("JSONL files", "*.jsonl")])
        if self.file_path:
            messagebox.showinfo("File Selected", f"File selected: {self.file_path}")

    def filter_articles(self):
        if not self.file_path:
            messagebox.showerror("No File", "Please select a file first.")
            return

        query = self.query_entry.get()
        if not query:
            messagebox.showerror("No Query", "Please enter a query.")
            return

        try:
            # Load the dataset
            data = pd.read_json(self.file_path, lines=True)
            data['content'] = data['content'].str.lower().str.replace(r'[^\w\s]', '', regex=True)
            data['title'] = data['title'].str.lower().str.replace(r'[^\w\s]', '', regex=True)
            data['source'] = data['source'].str.lower().str.replace(r'[^\w\s]', '', regex=True)
            data['published'] = pd.to_datetime(data['published'])

            # Prepare query and article embeddings
            reference_query = str(query)
            data['combined_text'] = data['title'] + ". " + data['content']
            article_embeddings = self.model.encode(data['combined_text'].tolist(), convert_to_tensor=True)
            query_embedding = self.model.encode(reference_query.lower(), convert_to_tensor=True)

            # Compute cosine similarity
            data['similarity'] = util.cos_sim(query_embedding, article_embeddings).squeeze().cpu().numpy()

            # Filter and sort by relevance
            self.filtered_articles = data[data['similarity'] > 0.5].sort_values(by='similarity', ascending=False)

            # Display results in the text box
            self.result_text.delete(1.0, tk.END)
            for _, row in self.filtered_articles.iterrows():
                self.result_text.insert(tk.END, f"Date: {row['published']}\n")
                self.result_text.insert(tk.END, f"Title: {row['title']}\n")
                self.result_text.insert(tk.END, f"Content: {row['content']}\n")
                self.result_text.insert(tk.END, "-"*80 + "\n")
            messagebox.showinfo("Filtering Done", "Filtering complete. Results are shown.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def save_results(self):
        if self.filtered_articles is None:
            messagebox.showerror("No Results", "No results to save. Please filter articles first.")
            return

        save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if save_path:
            self.filtered_articles[['id', 'title', 'content', 'published', 'similarity']].to_csv(save_path, index=False)
            messagebox.showinfo("Results Saved", f"Results saved to: {save_path}")

# Running the GUI application
if __name__ == "__main__":
    root = tk.Tk()
    app = NewsFilterGUI(root)
    root.mainloop()
