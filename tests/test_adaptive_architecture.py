import sys
from unittest.mock import MagicMock

# Mock numpy and torch before any ebaif imports
mock_np = MagicMock()
mock_np.bool_ = bool  # Fix pytest.approx TypeError
sys.modules['numpy'] = mock_np
sys.modules['torch'] = MagicMock()
sys.modules['torch.nn'] = MagicMock()
sys.modules['torch.optim'] = MagicMock()
sys.modules['src.ebaif.consensus.protocol'] = MagicMock()
sys.modules['src.ebaif.consensus.reputation'] = MagicMock()
sys.modules['ebaif.consensus.protocol'] = MagicMock()
sys.modules['ebaif.consensus.reputation'] = MagicMock()

import pytest
from src.ebaif.advanced.adaptive_architecture import NetworkNode

class TestNetworkNodeUpdateUtilization:
    """Tests for NetworkNode.update_utilization method."""

    def test_initial_state(self):
        """Verify initial state of a NetworkNode."""
        node = NetworkNode("test_node", 0)
        assert node.utilization_score == 0.0
        assert len(node.activation_history) == 0
        assert len(node.gradient_history) == 0

    def test_single_update(self):
        """Test a single update of utilization."""
        node = NetworkNode("test_node", 0)
        node.update_utilization(0.5, 0.3)

        # activation_history = [0.5], gradient_history = [0.3]
        # avg_activation = 0.5, avg_gradient = 0.3
        # utilization_score = (0.5 + 0.3) / 2 = 0.4
        assert node.utilization_score == pytest.approx(0.4)
        assert list(node.activation_history) == [0.5]
        assert list(node.gradient_history) == [0.3]

    def test_multiple_updates(self):
        """Test multiple updates to verify averaging logic."""
        node = NetworkNode("test_node", 0)

        # Update 1
        node.update_utilization(0.8, 0.2)
        assert node.utilization_score == pytest.approx(0.5)

        # Update 2
        node.update_utilization(0.4, 0.6)
        # avg_activation = (0.8 + 0.4) / 2 = 0.6
        # avg_gradient = (0.2 + 0.6) / 2 = 0.4
        # utilization_score = (0.6 + 0.4) / 2 = 0.5
        assert node.utilization_score == pytest.approx(0.5)

        # Update 3
        node.update_utilization(0.0, 1.0)
        # avg_activation = (0.8 + 0.4 + 0.0) / 3 = 0.4
        # avg_gradient = (0.2 + 0.6 + 1.0) / 3 = 0.6
        # utilization_score = (0.4 + 0.6) / 2 = 0.5
        assert node.utilization_score == pytest.approx(0.5)

    def test_absolute_value_handling(self):
        """Verify that negative values are converted to absolute values."""
        node = NetworkNode("test_node", 0)
        node.update_utilization(-0.6, -0.4)

        # Should be treated as 0.6 and 0.4
        assert node.utilization_score == pytest.approx(0.5)
        assert list(node.activation_history) == [0.6]
        assert list(node.gradient_history) == [0.4]

    def test_history_limit(self):
        """Verify that history respects the maxlen limit (100)."""
        node = NetworkNode("test_node", 0)

        # Fill history beyond maxlen
        for i in range(110):
            node.update_utilization(1.0, 1.0)

        assert len(node.activation_history) == 100
        assert len(node.gradient_history) == 100
        assert node.utilization_score == pytest.approx(1.0)
