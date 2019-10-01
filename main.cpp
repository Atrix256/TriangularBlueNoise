#define _CRT_SECURE_NO_WARNINGS

#include <random>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

static const size_t c_gradientWidth = 256;
static const size_t c_gradientHeight = 32;

template <typename T>
T Clamp(T value, T min, T max)
{
    if (value < min)
        return min;
    else if (value > max)
        return max;
    else
        return value;
}

float GradientValue(size_t pixelX, size_t pixelY)
{
    return float(pixelX) / float(c_gradientWidth - 1);
}

unsigned char GradientValueU8(size_t pixelX, size_t pixelY)
{
    float value = GradientValue(pixelX, pixelY);
    return (unsigned char)Clamp(value * 255.0f + 0.5f, 0.0f, 255.0f);
}

int main(int argc, char** argv)
{
    // init random number generator
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<float> dist;

    {
        std::vector<unsigned char> gradient(c_gradientWidth*c_gradientHeight);
        for (size_t iy = 0; iy < c_gradientHeight; ++iy)
            for (size_t ix = 0; ix < c_gradientWidth; ++ix)
                gradient[iy*c_gradientWidth + ix] = GradientValueU8(ix, iy);

        stbi_write_png("out_gradient.png", c_gradientWidth, c_gradientHeight, 1, gradient.data(), 0);
    }

    {
        std::vector<unsigned char> gradient(c_gradientWidth*c_gradientHeight);
        for (size_t iy = 0; iy < c_gradientHeight; ++iy)
            for (size_t ix = 0; ix < c_gradientWidth; ++ix)
                gradient[iy*c_gradientWidth + ix] = (GradientValue(ix, iy) > dist(rng)) ? 255 : 0;

        stbi_write_png("out_gradient_white.png", c_gradientWidth, c_gradientHeight, 1, gradient.data(), 0);
    }

    return 0;
}


/*

TODO:
* find "inside" that talks about triangular distributed noise
 * page 54 here. there's a link to a paper too. https://www.gdcvault.com/play/1023002/Low-Complexity-High-Fidelity-INSIDE
 * paper: https://uwspace.uwaterloo.ca/bitstream/handle/10012/3867/thesis.pdf;jsessionid=74681FAF2CA22E754C673E9A1E6957EC?sequence=1

- white noise dither a gradient.
- blue noise dither a gradient.
- triangular white/blue noise a gradient (how to get triangular blue noise? maybe just white first?)

- Triangular blue noise constructions:
1) make blue noise with void and cluster, and inverse CDF the values.
2) try averaging 2 blue noise textures
3) try subtracting 0.5, squaring (or similar) and adding 0.5 back in.

- histogram of noise
- DFT of noise

- threshold tests of blue noise along with histogram and DFT




! DFT circle by packing hexagons?

*/