Some of the techniques used to improve performance for naive large matrix multiplications from the flash attention paper are:
- use of tiling to break down extremely large matrices into smaller blocks for computation to reduce having to read an entire large matrix into memory all at once
- fuses operations such as softmax and weighted sum into a single kernel to reduce memory reads/writes
- manages how data is loaded and stored to minimize memory bandwidth bottlenecks

To adjust my current code with these optimizations:
- Implement tiling to allow the computing of dot-product attention scores on smaller sub-blocks
- Write a custom JAX function to fuse the softmax nd the weighted sum operations
- reorder the operations in the attention block to maximize cache reuse