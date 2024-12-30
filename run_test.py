import unittest

"""
Ce fichier permet de lancer l'ensemble des tests. Il faut s'assurer que tous les tests sont dans un 
dossier "tests", au format "test_*.py".
"""
def run_all_tests():
    # Détecte automatiquement tous les fichiers de tests dans le dossier "tests"
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir="tests", pattern="test_*.py")

    # Exécute les tests trouvés
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Affiche un résumé clair du résultat
    if result.wasSuccessful():
        print("Tous les tests ont été exécutés avec succès.")
        exit(0)
    else:
        print(f"{len(result.failures)} ont échoué.")
        exit(1)

if __name__ == "__main__":
    run_all_tests()
