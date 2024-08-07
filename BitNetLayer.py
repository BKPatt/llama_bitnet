import torch
import torch.nn as nn
from abc import ABC, abstractmethod

from BitNetConverter import BitNetConverter

class BitNetLayer(nn.Module, ABC):
    """
    Abstract base class for BitNet layers.
    """

    def __init__(self):
        super().__init__()
        self.scaling_factor = nn.Parameter(torch.Tensor([1.0]))

    @abstractmethod
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the layer.
        
        Args:
            input (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor.
        """
        pass

    @abstractmethod
    def quantize_weights(self):
        """
        Quantize the weights of the layer to binary values.
        """
        pass

    def update_scaling_factors(self):
        """
        Update the scaling factors of the layer.
        """
        if hasattr(self, 'weight'):
            with torch.no_grad():
                self.scaling_factor.data = self.weight.abs().mean()

    def apply_scaling(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply scaling factor to the tensor.
        
        Args:
            tensor (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Scaled tensor.
        """
        return BitNetConverter.apply_scaling_factor(tensor, self.scaling_factor)

    def binarize(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Binarize the input tensor to strictly -1 or 1.
        """
        return torch.sign(tensor)

    def stochastic_binarize(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Stochastically binarize the input tensor.
        
        Args:
            tensor (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Stochastically binarized tensor.
        """
        return BitNetConverter.stochastic_binarization(tensor)

    def straight_through(self, input: torch.Tensor, binarized_input: torch.Tensor) -> torch.Tensor:
        """
        Apply straight-through estimator.
        
        Args:
            input (torch.Tensor): Original input tensor.
            binarized_input (torch.Tensor): Binarized input tensor.
        
        Returns:
            torch.Tensor: Tensor for backpropagation.
        """
        return BitNetConverter.straight_through_estimator(input, binarized_input)