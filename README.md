# TriangularBlueNoise
Time to learn about triangular distributed blue noise

Alan: The remap to triangle doesn't strictly map [0,1] to [-0.5,1.5]. That is likely causing some issues. The below is better.
```cpp
// adapted from https://www.shadertoy.com/view/4t2SDh
float ReshapeUniformToTriangle(float rnd)
{
    rnd = frac(rnd + 0.5f);
    float orig = rnd * 2.0f - 1.0f;
    rnd = (orig == 0.0f) ? -1.0f : (orig / sqrt(abs(orig)));
    rnd = rnd - sign(orig) + 0.5f;
    return rnd;
}
```
