#define _CRT_SECURE_NO_WARNINGS

#include <random>
#include <vector>
#include <stdint.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

typedef uint8_t uint8;
typedef uint32_t uint32;



#define DETERMINISTIC() true  // if true, will use the seed below for everything, else will randomly generate a seed.
#define DETERMINISTIC_SEED() unsigned(783104853), unsigned(4213684301), unsigned(3526061164), unsigned(614346169), unsigned(478811579), unsigned(2044310268), unsigned(3671768129), unsigned(206439072)

static const size_t c_gradientWidth = 256;
static const size_t c_gradientHeight = 32;



inline std::seed_seq& GetRNGSeed()
{
#if DETERMINISTIC()
    static std::seed_seq fullSeed{ DETERMINISTIC_SEED() };
#else
    static std::random_device rd;
    static std::seed_seq fullSeed{ rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd() };
#endif
    return fullSeed;
}

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
uint8 GradientValueU8(size_t pixelX, size_t pixelY)
{
    float value = float(pixelX) / float(c_gradientWidth - 1);
    return (uint8)Clamp(value * 255.0f + 0.5f, 0.0f, 255.0f);
}

// Assumes src and dest have the same width
void AppendImageVertical(std::vector<uint8>& dest, const std::vector<uint8>& src)
{
    size_t destSize = dest.size();
    size_t srcSize = src.size();
    dest.resize(destSize + srcSize);
    memcpy(&dest[destSize], src.data(), srcSize);
}

template <typename LAMBDA>
void DoTest(const char* fileName, const std::vector<uint8>& gradient, const LAMBDA& lambda)
{
    std::vector<uint8> gradientDithered(c_gradientWidth*c_gradientHeight);
    std::vector<uint8> gradientError(c_gradientWidth*c_gradientHeight);
    for (size_t iy = 0; iy < c_gradientHeight; ++iy)
    {
        for (size_t ix = 0; ix < c_gradientWidth; ++ix)
        {
            uint8 randValue = lambda(ix, iy);
            gradientDithered[iy*c_gradientWidth + ix] = (gradient[iy*c_gradientWidth + ix] > randValue) ? 255 : 0;
            if (gradient[iy*c_gradientWidth + ix] >= randValue)
                gradientError[iy*c_gradientWidth + ix] = gradient[iy*c_gradientWidth + ix] - randValue;
            else
                gradientError[iy*c_gradientWidth + ix] = randValue - gradient[iy*c_gradientWidth + ix];
        }
    }

    std::vector<uint8> outImage = gradient;
    AppendImageVertical(outImage, gradientDithered);
    AppendImageVertical(outImage, gradientError);
    stbi_write_png(fileName, c_gradientWidth, c_gradientHeight * 3, 1, outImage.data(), 0);
}

int main(int argc, char** argv)
{
    // init random number generator
    std::mt19937 rng(GetRNGSeed());

    // make the gradient image
    std::vector<uint8> gradient(c_gradientWidth*c_gradientHeight);
    {
        for (size_t iy = 0; iy < c_gradientHeight; ++iy)
            for (size_t ix = 0; ix < c_gradientWidth; ++ix)
                gradient[iy*c_gradientWidth + ix] = GradientValueU8(ix, iy);
    }

    // uniform white noise test
    DoTest("out_gradient_white_uniform.png", gradient,
        [&rng] (size_t ix, size_t iy)
        {
            static std::uniform_int_distribution<uint32> dist(0, 255);
            return uint8(dist(rng));
        }
    );

    // triangular white noise test
    DoTest("out_gradient_white_triangle_avg.png", gradient,
        [&rng] (size_t ix, size_t iy)
        {
            static std::uniform_int_distribution<uint32> dist(0, 255);
            return uint8((dist(rng) + dist(rng))/2);
        }
    );

    // TODO: show an image for error, and another for histogram.
    // TODO: test multiple quantizations? not just 1 bit? i think thats where the triangular noise helps
    // TODO: inside used triangle noise that went outside 0 to 1. is that important?
    // TODO: try making triangle noise by inverting cdf.

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