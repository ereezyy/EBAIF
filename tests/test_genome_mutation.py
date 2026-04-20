import unittest
import torch
import pytest
from src.ebaif.behavior_genome.genome import BehaviorGenome, GenomeConfig

class TestGenomeMutation(unittest.TestCase):
    def setUp(self):
        self.config = GenomeConfig(
            mutation_probability=0.1,
            input_dim=10,
            output_dim=2,
            hidden_dims=[16]
        )
        self.genome = BehaviorGenome(self.config)

    def test_mutate_creates_new_instance(self):
        """Test that mutate returns a new instance, not the same object."""
        mutated = self.genome.mutate()
        self.assertNotEqual(id(self.genome), id(mutated))
        self.assertNotEqual(self.genome.genome_id, mutated.genome_id)
        self.assertEqual(mutated.generation, self.genome.generation + 1)
        self.assertEqual(mutated.parent_ids, [self.genome.genome_id])

    def test_mutate_modifies_genes_high_rate(self):
        """Test that genes are modified when mutation rate is high."""
        # Force high mutation rate to ensure changes
        mutated = self.genome.mutate(mutation_rate=1.0)

        # Check behavior genes
        behavior_changed = False
        for key, gene in self.genome.behavior_genes.items():
            if not torch.equal(gene, mutated.behavior_genes[key]):
                behavior_changed = True
                break

        # Check architecture genes
        architecture_changed = False
        for key, gene in self.genome.architecture_genes.items():
            # Skip if it's the activation vector because random choice might pick the same one
            if key == 'activation':
                continue
            if not torch.equal(gene, mutated.architecture_genes[key]):
                architecture_changed = True
                break

        # At least some genes should change with 1.0 mutation rate
        self.assertTrue(behavior_changed or architecture_changed, "Mutation with rate 1.0 did not change any genes")

    def test_mutate_bounds(self):
        """Test that mutated values stay within expected bounds."""
        mutated = self.genome.mutate(mutation_rate=1.0)

        # Check behavior genes (0.0 to 1.0)
        for key, gene in mutated.behavior_genes.items():
            self.assertTrue(torch.all(gene >= 0.0), f"Behavior gene {key} < 0.0")
            self.assertTrue(torch.all(gene <= 1.0), f"Behavior gene {key} > 1.0")

        # Check architecture genes (0.001 to 10.0 for continuous)
        for key, gene in mutated.architecture_genes.items():
            if gene.is_floating_point() and key != 'activation':
                 self.assertTrue(torch.all(gene >= 0.001), f"Architecture gene {key} < 0.001")
                 self.assertTrue(torch.all(gene <= 10.0), f"Architecture gene {key} > 10.0")

    def test_mutate_zero_rate(self):
        """Test that zero mutation rate preserves genes exactly."""
        mutated = self.genome.mutate(mutation_rate=0.0)

        # Check behavior genes
        for key, gene in self.genome.behavior_genes.items():
            self.assertTrue(torch.equal(gene, mutated.behavior_genes[key]))

        # Check architecture genes
        for key, gene in self.genome.architecture_genes.items():
             self.assertTrue(torch.equal(gene, mutated.architecture_genes[key]))

if __name__ == '__main__':
    unittest.main()
