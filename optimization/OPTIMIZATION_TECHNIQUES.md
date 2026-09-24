CNNs have a lot of exploitable structure
Convolution has very strong mathematical structure:
$$ K_h \times K_w \times C_{in} \times C_{out} $$
The same kernel is repeatedly applied over:
$$ H\times W $$
positions.
That creates many dimensions researchers can manipulate.


| Optimization                | Exploits                           |
| --------------------------- | ---------------------------------- |
| Depthwise convolution       | channel independence               |
| Group convolution           | channel grouping                   |
| Pointwise \(1\times1\) conv | spatial simplification             |
| Kernel pruning              | kernel redundancy                  |
| Channel pruning             | feature-map redundancy             |
| Filter pruning              | output-channel redundancy          |
| Winograd                    | repeated small convolutions        |
| FFT convolution             | convolution mathematical structure |
| Low-rank decomposition      | tensor redundancy                  |
| Kernel fusion               | neighboring operations             |
| Tiling                      | spatial locality                   |
