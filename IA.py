import os
import sys
import docx
import pdfplumber
from bs4 import BeautifulSoup
import markdown
from transformers import pipeline
from PyQt6.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QTextEdit,
    QFileDialog, QMessageBox
)

# Fonction pour lire fichier txt, pdf, docx, html, md
def lire_fichier(chemin):
    ext = os.path.splitext(chemin)[1].lower()

    if ext == ".txt":
        with open(chemin, "r", encoding="utf-8") as f:
            return f.read()
    elif ext == ".pdf":
        texte = ""
        with pdfplumber.open(chemin) as pdf:
            for page in pdf.pages:
                texte += page.extract_text() or ''
        return texte
    elif ext == ".docx":
        doc = docx.Document(chemin)
        return "\n".join(p.text for p in doc.paragraphs)
    elif ext == ".html" or ext == ".htm":
        with open(chemin, "r", encoding="utf-8") as f:
            html_content = f.read()
        soup = BeautifulSoup(html_content, 'html.parser')
        return soup.get_text(separator="\n")
    elif ext == ".md":
        with open(chemin, "r", encoding="utf-8") as f:
            md_content = f.read()
        html = markdown.markdown(md_content)
        soup = BeautifulSoup(html, 'html.parser')
        return soup.get_text(separator="\n")
    else:
        return ""

# Fonction pour découper un texte en chunks (listes de sous-textes)
def decouper_texte(texte, taille_max=500):
    mots = texte.split()
    chunks = []
    for i in range(0, len(mots), taille_max):
        chunk = " ".join(mots[i:i+taille_max])
        chunks.append(chunk)
    return chunks

# Pipelines
resumeur = pipeline("summarization", model="t5-small", tokenizer="t5-small", framework="pt")
classifieur = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

themes = ["politique", "philosophie", "sport", "économie", "technologie", "santé", "environnement", "culture", "histoire"]

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Résumé + Détection de thème améliorés")
        self.setGeometry(100, 100, 700, 600)

        self.layout = QVBoxLayout()

        self.btn_choisir = QPushButton("Choisir un fichier et lancer l'analyse")
        self.btn_choisir.clicked.connect(self.lancer_analyse)
        self.layout.addWidget(self.btn_choisir)

        self.txt_resultat = QTextEdit()
        self.txt_resultat.setReadOnly(True)
        self.layout.addWidget(self.txt_resultat)

        self.setLayout(self.layout)

    def lancer_analyse(self):
        chemin_fichier, _ = QFileDialog.getOpenFileName(
            self,
            "Choisir un fichier",
            "",
            "Tous fichiers supportés (*.txt *.pdf *.docx *.html *.htm *.md);;"
            "Texte (*.txt);;PDF (*.pdf);;Word (*.docx);;HTML (*.html *.htm);;Markdown (*.md)"
        )
        if not chemin_fichier:
            return

        self.txt_resultat.setText("Analyse en cours, veuillez patienter...")

        try:
            texte = lire_fichier(chemin_fichier)
            if not texte.strip():
                QMessageBox.critical(self, "Erreur", "Fichier vide ou format non supporté.")
                self.txt_resultat.clear()
                return

            # Découper texte en chunks
            chunks = decouper_texte(texte, taille_max=500)

            # Résumer chaque chunk puis agréger
            resumes = []
            for chunk in chunks:
                res = resumeur("summarize: " + chunk, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
                resumes.append(res)

            resume_final = " ".join(resumes)

            # Classification sur le texte complet (ou résumé agrégé)
            classification = classifieur(texte, candidate_labels=themes)
            theme_principal = classification['labels'][0]

            texte_affiche = (
                f"Thème détecté : {theme_principal}\n\n"
                f"Résumé complet :\n{resume_final}"
            )

            self.txt_resultat.setText(texte_affiche)

        except Exception as e:
            QMessageBox.critical(self, "Erreur", str(e))
            self.txt_resultat.clear()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    fenetre = App()
    fenetre.show()
    sys.exit(app.exec())
