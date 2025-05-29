"""
###LIGNE 246 POUR LANGUE ET LIGNE 207 POUR CHEMIN D'ACCES A L'IMAGE###
Script de test pour le système OCR Enhanced
Test spécifique pour le fichier lorem ipsum
Version mise à jour pour CodeSourceOCR_EN_FR.py - CORRIGÉE
"""

import sys
import os
from pathlib import Path

"""
SOLUTION RAPIDE POPPLER - À ajouter au début de votre script OCR
"""

import os
import sys

def fix_poppler_path():
    """
    Correction automatique du PATH Poppler pour la session Python actuelle
    """
    # Chemins possibles pour Poppler (ajustez selon votre installation)
    possible_paths = [
        r"C:\Users\bauer\Downloads\Release-24.08.0-0 (1)\poppler-24.08.0\Library\bin",
        r"C:\Users\bauer\Downloads\Release-24.08.0-0 (1)\poppler-24.08.0\bin",
        r"C:\poppler\bin",
        r"C:\Program Files\poppler\bin",
        r"C:\poppler-24.08.0\bin"
    ]
    
    # Trouver le bon chemin
    poppler_path = None
    for path in possible_paths:
        if os.path.exists(path) and os.path.exists(os.path.join(path, 'pdfinfo.exe')):
            poppler_path = path
            print(f"📁 Poppler trouvé: {poppler_path}")
            break
    
    if not poppler_path:
        print("❌ Erreur: Aucun dossier Poppler valide trouvé")
        print("Chemins vérifiés:")
        for path in possible_paths:
            print(f"   - {path}")
        return False
    
    # Vérifier que les fichiers essentiels sont présents
    required_files = ['pdfinfo.exe', 'pdftoppm.exe']
    for file in required_files:
        if not os.path.exists(os.path.join(poppler_path, file)):
            print(f"❌ Erreur: {file} manquant dans {poppler_path}")
            return False
    
    # Ajouter au PATH de la session actuelle
    current_path = os.environ.get('PATH', '')
    if poppler_path not in current_path:
        os.environ['PATH'] = poppler_path + os.pathsep + current_path
        print(f"✅ Poppler ajouté au PATH: {poppler_path}")
    else:
        print(f"✅ Poppler déjà dans le PATH")
    
    return True

def test_poppler():
    """
    Test pour vérifier que Poppler fonctionne
    """
    import subprocess
    try:
        result = subprocess.run(['pdfinfo', '-v'], 
                               capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ Poppler fonctionne! Version: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ Erreur Poppler: {result.stderr}")
            return False
    except FileNotFoundError:
        print("❌ pdfinfo introuvable - PATH non configuré")
        return False
    except Exception as e:
        print(f"❌ Erreur test Poppler: {e}")
        return False

# Configuration initiale Poppler
print("🔧 Configuration Poppler...")
if fix_poppler_path():
    if test_poppler():
        print("🎉 Poppler configuré avec succès!")
    else:
        print("⚠️ Poppler partiellement configuré")
else:
    print("⚠️ Configuration Poppler échouée")

# Import du nouveau système OCR Enhanced - CORRIGÉ
try:
    # Import depuis le nouveau fichier CodeSourceOCR_EN_FR.py
    from CodeSourceOCR_EN_FR import (
    EnhancedOCRPipeline,  # ← CORRIGÉ: Nouveau nom
    OCRConfig, 
    create_optimized_config_for_lorem_ipsum,
    create_universal_config,  # ← AJOUT: Nouvelle fonction
    AdvancedImagePreprocessor,
    EnhancedOCREngine  # ← CORRIGÉ: Nouveau nom
)
    print("✅ Import du système OCR Enhanced réussi")
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    print("Vérifiez que le fichier CodeSourceOCR_EN_FR.py est dans le même dossier")
    print("Et que toutes les dépendances sont installées:")
    print("pip install opencv-python pytesseract Pillow numpy matplotlib pdf2image python-Levenshtein tqdm easyocr")
    sys.exit(1)

def test_installation():
    """Test de l'installation des composants"""
    print("\n🧪 Test de l'installation...")
    
    components_ok = True
    
    try:
        import cv2
        print("✅ OpenCV disponible")
    except ImportError:
        print("❌ OpenCV manquant")
        components_ok = False
    
    try:
        import pytesseract
        version = pytesseract.get_tesseract_version()
        print(f"✅ Tesseract disponible (version: {version})")
    except Exception as e:
        print(f"❌ Problème Tesseract: {e}")
        components_ok = False
    
    try:
        from pdf2image import convert_from_path
        print("✅ pdf2image disponible")
    except ImportError:
        print("❌ pdf2image manquant")
        components_ok = False
    
    try:
        import easyocr
        print("✅ EasyOCR disponible")
    except ImportError:
        print("⚠️ EasyOCR manquant (recommandé mais optionnel)")
    
    try:
        import paddleocr
        # Test d'import complet pour éviter l'erreur paddlex
        from paddleocr import PaddleOCR
        print("✅ PaddleOCR disponible")
    except (ImportError, ModuleNotFoundError) as e:
        print(f"⚠️ PaddleOCR manquant ou incomplet ({e}) - optionnel")
    
    return components_ok

def test_ocr_simple():
    """Test OCR simple avec image de test"""
    print("\n🔍 Test OCR simple...")
    
    try:
        import numpy as np
        import cv2
        
        # Création d'une image de test simple avec "Lorem Ipsum"
        test_img = np.zeros((150, 500, 3), dtype=np.uint8)
        test_img.fill(255)  # Fond blanc
        
        # Ajout de texte de test
        cv2.putText(test_img, "Lorem Ipsum", (50, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        cv2.putText(test_img, "dolor sit amet", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Configuration optimisée pour le test
        config = create_optimized_config_for_lorem_ipsum()
        config.save_debug_images = False  # Pas de debug pour le test simple
        
        pipeline = EnhancedOCRPipeline(config)  # ← CORRIGÉ: Nouveau nom
        
        results = pipeline._process_image_array(test_img, "test_simple")
        
        extracted_text = results.get('text', '').lower()
        
        if "lorem" in extracted_text and "ipsum" in extracted_text:
            print(f"✅ Test OCR réussi: '{results['text'].strip()}'")
            return True
        elif len(extracted_text.strip()) > 0:
            print(f"⚠️ Test OCR partiel: '{results['text'].strip()}'")
            print("Le texte est détecté mais peut nécessiter des ajustements")
            return True
        else:
            print(f"❌ Test OCR échoué: texte vide")
            return False
            
    except Exception as e:
        print(f"❌ Erreur test OCR: {e}")
        import traceback
        traceback.print_exc()
        return False

def process_lorem_ipsum_file():
    """Traitement principal du fichier lorem ipsum"""
    # Chemins possibles pour votre fichier
    possible_files = [
        r"C:\Users\bauer\Downloads\thaitest.jpeg",
        r"C:\Users\bauer\Downloads\lorem-ipsum-meaning-in-english-lipsumhub.jpg",
        r"C:\Users\bauer\Downloads\imtest.pdf",
        r"lorem_ipsum.jpg",
        r"imtest.pdf"
    ]
    
    file_path = None
    for path in possible_files:
        if os.path.exists(path):
            file_path = path
            break
    
    if not file_path:
        print("❌ Aucun fichier lorem ipsum trouvé dans les chemins suivants:")
        for path in possible_files:
            print(f"   - {path}")
        print("\nVeuillez vérifier le chemin de votre fichier.")
        return False
    
    output_dir = r"C:\Users\bauer\Downloads\ocr_results_enhanced"
    
    print(f"\n📄 Traitement de: {file_path}")
    
    # Création du dossier de sortie
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Configuration universelle adaptative
        config = OCRConfig(
            # Prétraitement adapté au thaï
            enable_preprocessing=True,
            enable_deskewing=True,
            enable_denoising=True,
            enable_sharpening=False,  # Peut dégrader les caractères thaï
            enable_contrast_enhancement=True,
            enable_adaptive_threshold=True,
            scale_factor=3.0,  # Réduit un peu
            
            # Configuration OCR pour thaï uniquement
            language='tha',
            tesseract_config='--psm 3 -c preserve_interword_spaces=1',  # Préserve les espaces
            confidence_threshold=1.0,  # Très très bas
            enable_auto_language_detection=False,
            
            # EasyOCR prioritaire pour le thaï
            enable_fallback=True,
            fallback_engines=['easyocr'],
            max_retry_attempts=1,
            
            # Post-traitement totalement désactivé
            enable_text_cleaning=False,
            enable_spell_correction=False,
            enable_entity_extraction=False,
            enable_structure_analysis=False,
            
            # Performance
            enable_caching=False,
            enable_parallel_processing=False,
            
            # Sortie avec debug
            save_json=True,
            save_visualization=True,
            save_debug_images=True,
            output_dir=output_dir
        )
        
        print("⚙️ Configuration optimisée Lorem Ipsum:")
        print(f"   - Langues: {config.language}")
        print(f"   - Seuil confiance: {config.confidence_threshold}%")
        print(f"   - Facteur d'échelle: {config.scale_factor}x")
        print(f"   - Prétraitement: {'Activé' if config.enable_preprocessing else 'Désactivé'}")
        print(f"   - Fallback: {'Activé' if config.enable_fallback else 'Désactivé'}")
        print(f"   - Debug images: {'Activé' if config.save_debug_images else 'Désactivé'}")
        
        # Initialisation du pipeline
        pipeline = EnhancedOCRPipeline(config)  # ← CORRIGÉ: Nouveau nom
        
        print("\n🚀 Début du traitement...")
        
        # Déterminer le type de fichier et traiter
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.pdf':
            print("📄 Traitement PDF...")
            results = pipeline.process_pdf_file(file_path)
        elif file_ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
            print("🖼️ Traitement image...")
            results = pipeline.process_image_file(file_path)
        else:
            print(f"❌ Format non supporté: {file_ext}")
            return False
        
        # Sauvegarde des résultats
        output_name = f"lorem_ipsum_results_{Path(file_path).stem}"
        output_path = os.path.join(output_dir, output_name)
        pipeline.save_results_to_files(results, output_path)
        
        # Affichage des résultats
        print("\n" + "="*70)
        print("📊 RÉSULTATS DU TRAITEMENT LOREM IPSUM")
        print("="*70)
        
        if results.get('success', False):
            print("✅ Extraction réussie!")
            
            # Informations de base
            processing_time = results.get('processing_time', 0)
            print(f"⏱️ Temps de traitement: {processing_time:.2f} secondes")
            
            # Pour les PDFs
            if 'total_pages' in results:
                print(f"📄 Pages traitées: {results.get('successful_pages', 0)}/{results.get('total_pages', 0)}")
            
            # Statistiques de caractères et mots
            char_count = results.get('character_count', len(results.get('text', '')))
            word_count = results.get('word_count', len(results.get('text', '').split()))
            print(f"📝 Caractères extraits: {char_count}")
            print(f"📝 Mots extraits: {word_count}")
            
            # Confiance
            if 'confidence_metrics' in results:
                conf = results['confidence_metrics']
                avg_conf = conf.get('average', 0)
                min_conf = conf.get('minimum', 0)
                max_conf = conf.get('maximum', 0)
                print(f"📊 Confiance moyenne: {avg_conf:.1f}%")
                print(f"📈 Confiance min/max: {min_conf:.1f}% / {max_conf:.1f}%")
            
            # Performance des moteurs
            if 'engine_performance' in results:
                engine_perf = results['engine_performance']
                if engine_perf:
                    print(f"\n🔧 PERFORMANCE DES MOTEURS:")
                    for engine, stats in engine_perf.items():
                        success_rate = stats.get('success_rate', 0)
                        attempts = stats.get('total_attempts', 0)
                        print(f"   {engine.capitalize()}: {success_rate:.1f}% ({attempts} tentatives)")
            
            # Analyse spécifique Lorem Ipsum
            extracted_text = results.get('text', '').lower()
            lorem_words = ['lorem', 'ipsum', 'dolor', 'sit', 'amet', 'consectetur', 'adipiscing', 'elit']
            found_words = [word for word in lorem_words if word in extracted_text]
            
            print(f"\n🎯 ANALYSE LOREM IPSUM:")
            print(f"   Mots Lorem détectés: {len(found_words)}/{len(lorem_words)}")
            if found_words:
                print(f"   Mots trouvés: {', '.join(found_words)}")
            else:
                print("   ⚠️ Aucun mot Lorem Ipsum standard détecté")
            
            # Aperçu du texte
            full_text = results.get('text', '')
            if full_text:
                preview_length = 300
                preview = full_text[:preview_length]
                
                print(f"\n📝 TEXTE EXTRAIT ({len(full_text)} caractères total):")
                print("-" * 50)
                print(preview.strip())
                if len(full_text) > preview_length:
                    print("...")
                
                # Analyse des problèmes potentiels
                if "meaning" in full_text.lower() and "lorem" not in full_text.lower():
                    print("\n🚨 PROBLÈME DÉTECTÉ:")
                    print("   Le texte contient 'meaning' mais pas 'lorem'")
                    print("   Cela suggère une mauvaise reconnaissance OCR")
                    print("   Essayez avec des paramètres différents ou vérifiez la qualité de l'image")
                
            else:
                print("\n⚠️ Aucun texte extrait")
        else:
            print("❌ Échec de l'extraction")
            error_msg = results.get('error', 'Erreur inconnue')
            print(f"🚫 Erreur: {error_msg}")
        
        # Localisation des fichiers de sortie
        print(f"\n💾 FICHIERS DE SORTIE:")
        print(f"   📄 Texte: {output_path}.txt")
        if config.save_json:
            print(f"   📋 JSON: {output_path}.json")
        if config.save_debug_images:
            print(f"   🖼️ Images debug: {output_dir}/debug_images/")
        
        # Statistiques globales du pipeline
        global_stats = pipeline.get_global_statistics()
        if global_stats['files_processed'] > 0:
            print(f"\n📊 STATISTIQUES GLOBALES:")
            print(f"   Fichiers traités: {global_stats['files_processed']}")
            print(f"   Taux de réussite: {global_stats['success_rate']:.1f}%")
            print(f"   Caractères totaux: {global_stats['total_characters_extracted']}")
        
        print("\n" + "="*70)
        if results.get('success', False) and len(results.get('text', '').strip()) > 0:
            print("✅ TRAITEMENT TERMINÉ AVEC SUCCÈS")
        else:
            print("⚠️ TRAITEMENT TERMINÉ AVEC PROBLÈMES")
        print("="*70)
        
        return results.get('success', False)
        
    except Exception as e:
        print(f"\n❌ ERREUR lors du traitement: {e}")
        print("\nConseils de dépannage:")
        print("1. Vérifiez que Tesseract est installé et configuré")
        print("2. Vérifiez que Poppler est installé pour les PDFs")
        print("3. Vérifiez que le fichier n'est pas corrompu")
        print("4. Essayez le mode debug pour voir les étapes de prétraitement")
        print("5. Vérifiez la qualité de l'image source")
        
        import traceback
        print(f"\nDétails de l'erreur:")
        traceback.print_exc()
        
        return False

def diagnostic_avance():
    """Diagnostic avancé du système"""
    print("\n🔬 DIAGNOSTIC AVANCÉ DU SYSTÈME")
    print("="*50)
    
    # Test des composants individuels
    try:
        from CodeSourceOCR_EN_FR import (
            TESSERACT_AVAILABLE, 
            EASYOCR_AVAILABLE, 
            PADDLEOCR_AVAILABLE, 
            POPPLER_PATH
        )
        
        print("🔧 ÉTAT DES COMPOSANTS:")
        print(f"   Tesseract: {'✅ Disponible' if TESSERACT_AVAILABLE else '❌ Non disponible'}")
        print(f"   EasyOCR: {'✅ Disponible' if EASYOCR_AVAILABLE else '❌ Non disponible'}")
        print(f"   PaddleOCR: {'✅ Disponible' if PADDLEOCR_AVAILABLE else '❌ Non disponible'}")
        print(f"   Poppler: {'✅ Disponible' if POPPLER_PATH else '❌ Non disponible'}")
        
        if POPPLER_PATH:
            print(f"   Chemin Poppler: {POPPLER_PATH}")
        
    except Exception as e:
        print(f"❌ Erreur diagnostic: {e}")
    
    # SUPPRIMÉ: Test de qualité d'image car ImageQualityAssessment n'existe plus
    print(f"\n📊 TEST QUALITÉ D'IMAGE:")
    print("   Module de test de qualité non disponible dans cette version")

def main():
    """Fonction principale"""
    print("🚀 SYSTÈME OCR ENHANCED - TEST LOREM IPSUM")
    print("="*60)
    
    # Test de l'installation
    print("Phase 1: Vérification des composants...")
    if not test_installation():
        print("\n❌ Installation incomplète. Veuillez installer les dépendances manquantes.")
        print("\nCommandes d'installation:")
        print("pip install opencv-python pytesseract Pillow numpy matplotlib")
        print("pip install pdf2image python-Levenshtein tqdm")
        print("pip install easyocr")
        return 1
    
    # Test OCR simple
    print("\nPhase 2: Test OCR basique...")
    if not test_ocr_simple():
        print("\n❌ Test OCR échoué. Vérifiez la configuration de Tesseract.")
        print("Assurez-vous que Tesseract est installé et accessible.")
        return 1
    
    # Diagnostic avancé
    print("\nPhase 3: Diagnostic système...")
    diagnostic_avance()
    
    # Traitement du fichier principal
    print("\nPhase 4: Traitement du fichier Lorem Ipsum...")
    if process_lorem_ipsum_file():
        print("\n🎉 TOUS LES TESTS SONT RÉUSSIS!")
        print("Le système OCR Enhanced fonctionne correctement.")
        return 0
    else:
        print("\n⚠️ PROBLÈMES DÉTECTÉS LORS DU TRAITEMENT")
        print("Consultez les messages d'erreur ci-dessus pour le dépannage.")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        input("\nAppuyez sur Entrée pour fermer...")
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⏹️ Traitement interrompu par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()
        input("\nAppuyez sur Entrée pour fermer...")
        sys.exit(1)