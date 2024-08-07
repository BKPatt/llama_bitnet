import torch

class BitNetConverter:
    @staticmethod
    def quantize_to_bitnet(tensor: torch.Tensor, threshold: float = 0) -> torch.Tensor:
        """
        Quantize a floating-point tensor to binary (-1 or 1).
        
        Args:
            tensor (torch.Tensor): Input tensor to be quantized.
            threshold (float): Threshold for binarization. Default is 0.
        
        Returns:
            torch.Tensor: Binarized tensor.
        """
        return torch.sign(tensor - threshold)

    @staticmethod
    def bitnet_matmul(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        Perform matrix multiplication with a binarized weight matrix.
        
        Args:
            input (torch.Tensor): Input tensor.
            weight (torch.Tensor): Binarized weight tensor.
        
        Returns:
            torch.Tensor: Result of the matrix multiplication.
        """
        return torch.matmul(input, weight)

    @staticmethod
    def bitnet_convolution(input: torch.Tensor, weight: torch.Tensor, 
                           stride: int = 1, padding: int = 0, 
                           dilation: int = 1, groups: int = 1) -> torch.Tensor:
        """
        Perform convolution with a binarized weight tensor.
        
        Args:
            input (torch.Tensor): Input tensor.
            weight (torch.Tensor): Binarized weight tensor.
            stride (int): Convolution stride. Default is 1.
            padding (int): Convolution padding. Default is 0.
            dilation (int): Convolution dilation. Default is 1.
            groups (int): Number of groups for grouped convolution. Default is 1.
        
        Returns:
            torch.Tensor: Result of the convolution operation.
        """
        return torch.nn.functional.conv2d(input, weight, stride=stride, padding=padding, 
                                          dilation=dilation, groups=groups)

    @staticmethod
    def apply_scaling_factor(tensor: torch.Tensor, factor: float) -> torch.Tensor:
        """
        Apply a scaling factor to a tensor.
        
        Args:
            tensor (torch.Tensor): Input tensor.
            factor (float): Scaling factor to be applied.
        
        Returns:
            torch.Tensor: Scaled tensor.
        """
        return tensor * factor

    @staticmethod
    def stochastic_binarization(tensor: torch.Tensor) -> torch.Tensor:
        """
        Perform stochastic binarization on a tensor.
        
        Args:
            tensor (torch.Tensor): Input tensor to be binarized.
        
        Returns:
            torch.Tensor: Stochastically binarized tensor.
        """
        prob = (tensor + 1) / 2
        return (torch.rand_like(tensor) < prob).float() * 2 - 1

    @staticmethod
    def straight_through_estimator(input: torch.Tensor, 
                                   binarized_input: torch.Tensor) -> torch.Tensor:
        """
        Apply the straight-through estimator for backpropagation through 
        binarized values.
        
        Args:
            input (torch.Tensor): Original input tensor.
            binarized_input (torch.Tensor): Binarized input tensor.
        
        Returns:
            torch.Tensor: Tensor for backpropagation.
        """
        return (binarized_input - input).detach() + input