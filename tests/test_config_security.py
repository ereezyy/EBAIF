import unittest
import os
import sys
import shutil

# Add src to python path
sys.path.append(os.path.join(os.getcwd(), 'src'))

# We need to import Config without triggering ebaif.__init__ because of missing torch
# So we add src/ebaif/utils to sys.path and import config directly
sys.path.append(os.path.join(os.getcwd(), 'src/ebaif/utils'))
import config
Config = config.Config

class TestConfigSecurity(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.test_dir = 'test_output'
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_save_valid_json(self):
        """Test saving a valid JSON file."""
        filepath = os.path.join(self.test_dir, 'config.json')
        try:
            self.config.save_to_file(filepath)
            self.assertTrue(os.path.exists(filepath))
        except ValueError as e:
            self.fail(f"Valid save failed: {e}")

    def test_save_invalid_extension(self):
        """Test saving with invalid extension raises ValueError."""
        filepath = os.path.join(self.test_dir, 'config.txt')
        with self.assertRaises(ValueError):
            self.config.save_to_file(filepath)

    def test_save_traversal(self):
        """Test saving with traversal path raises ValueError."""
        # Try to save to parent directory
        filepath = '../test_config_traversal.json'
        with self.assertRaises(ValueError):
            self.config.save_to_file(filepath)

    def test_save_absolute_outside_cwd(self):
        """Test saving with absolute path outside CWD raises ValueError."""
        # Use /tmp for testing absolute path restriction
        filepath = '/tmp/test_config_absolute.json'
        with self.assertRaises(ValueError):
            self.config.save_to_file(filepath)

if __name__ == '__main__':
    unittest.main()
