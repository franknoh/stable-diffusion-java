# stable-diffusion-java
 
### how it works
this repo uses [DeepJavaLibrary](https://djl.ai) to run torchscript models on java.
the tokenizer and klms sampler were ported from [stable-diffusion-pytorch](https://github.com/kjsman/stable-diffusion-pytorch) with minimal changes.

### how to use
1. Install python
2. Install torch (see [here](https://pytorch.org/get-started/locally/))
3. Update pom.xml (see [here](https://docs.djl.ai/engines/pytorch/pytorch-engine/index.html))
4. Convert model to torchscript (see [here](https://github.com/franknoh/stable-diffusion-jit))
5. Run the code