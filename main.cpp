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

template <typename T>
T sign(T value)
{
    if (value < T(0))
        return T(-1);
    else if (value == T(0))
        return T(0);
    else
        return T(1);
}

// sRGB routines from https://www.nayuki.io/res/srgb-transform-library/srgb-transform.c
float srgb_to_linear_float(float x) {
    if (x <= 0.0f)
        return 0.0f;
    else if (x >= 1.0f)
        return 1.0f;
    else if (x < 0.04045f)
        return x / 12.92f;
    else
        return powf((x + 0.055f) / 1.055f, 2.4f);
}

// sRGB routines from https://www.nayuki.io/res/srgb-transform-library/srgb-transform.c
float linear_to_srgb_float(float x) {
    if (x <= 0.0f)
        return 0.0f;
    else if (x >= 1.0f)
        return 1.0f;
    else if (x < 0.0031308f)
        return x * 12.92f;
    else
        return powf(x, 1.0f / 2.4f) * 1.055f - 0.055f;
}

// Assumes src and dest have the same width
void AppendImageVertical(std::vector<float>& dest, const std::vector<float>& src)
{
    size_t destSize = dest.size();
    size_t srcSize = src.size();
    dest.resize(destSize + srcSize);
    memcpy(&dest[destSize], src.data(), srcSize * sizeof(src[0]));
}

void SaveImage(const char* fileName, const std::vector<float>& image, size_t width, size_t height)
{
    std::vector<uint8> outImage(image.size());
    for (size_t index = 0; index < image.size(); ++index)
    {
        float sRGB = linear_to_srgb_float(image[index]);
        outImage[index] = uint8(Clamp(sRGB * 255.0f + 0.5f, 0.0f, 255.0f));
    }

    stbi_write_png(fileName, int(width), int(height), 1, outImage.data(), 0);
}

template <typename LAMBDA>
void DoTest(const char* fileName, const std::vector<float>& gradient, const LAMBDA& lambda)
{
    std::vector<float> gradientDithered(c_gradientWidth*c_gradientHeight);
    std::vector<float> gradientError(c_gradientWidth*c_gradientHeight);
    std::vector<float> gradientHistogram(c_gradientWidth*c_gradientHeight);
    std::vector<uint32> histogram(c_gradientWidth, 0);

    for (size_t iy = 0; iy < c_gradientHeight; ++iy)
    {
        for (size_t ix = 0; ix < c_gradientWidth; ++ix)
        {
            // get random value
            float randValue = lambda(ix, iy);

            // dither
            gradientDithered[iy*c_gradientWidth + ix] = (gradient[iy*c_gradientWidth + ix] > randValue) ? 1.0f : 0.0f;

            // error image
            gradientError[iy*c_gradientWidth + ix] = std::abs(gradient[iy*c_gradientWidth + ix] - randValue);

            // histogram
            size_t histogramBucket = Clamp<size_t>(size_t(randValue * (c_gradientWidth - 1) + 0.5f), 0, c_gradientWidth - 1);
            histogram[histogramBucket]++;
        }
    }

    // make histogram image
    uint32 maxHistogramValue = histogram[0];
    for (uint32 histogramValue : histogram)
        maxHistogramValue = std::max(maxHistogramValue, histogramValue);

    for (size_t iy = 0; iy < c_gradientHeight; ++iy)
    {
        for (size_t ix = 0; ix < c_gradientWidth; ++ix)
        {
            float normalizedHistogramValue = 1.0f - float(histogram[ix]) / float(maxHistogramValue);
            size_t height = Clamp<size_t>(size_t(normalizedHistogramValue * (c_gradientHeight - 1) + 0.5), 0, c_gradientHeight-1);
            gradientHistogram[iy*c_gradientWidth + ix] = (iy < height) ? 1.0f : 0.0f;
        }
    }

    std::vector<float> outImage = gradient;
    AppendImageVertical(outImage, gradientDithered);
    AppendImageVertical(outImage, gradientError);
    AppendImageVertical(outImage, gradientHistogram);
    SaveImage(fileName, outImage, c_gradientWidth, c_gradientHeight * 4);
}

int main(int argc, char** argv)
{
    // init random number generator
    std::mt19937 rng(GetRNGSeed());

    // make the (linear) gradient image
    std::vector<float> gradient(c_gradientWidth*c_gradientHeight);
    {
        for (size_t iy = 0; iy < c_gradientHeight; ++iy)
            for (size_t ix = 0; ix < c_gradientWidth; ++ix)
                gradient[iy*c_gradientWidth + ix] = float(ix) / float(c_gradientWidth - 1);
    }

    // uniform white noise test
    DoTest("out_gradient_white_uniform.png", gradient,
        [&rng] (size_t ix, size_t iy)
        {
            static std::uniform_real_distribution<float> dist;
            return dist(rng);
        }
    );

    // triangular white noise test, made by averaging two white noise values
    DoTest("out_gradient_white_triangle_avg.png", gradient,
        [&rng] (size_t ix, size_t iy)
        {
            static std::uniform_real_distribution<float> dist;
            return (dist(rng) + dist(rng)) / 2;
        }
    );

    // triangular white noise test, made by reshaping a single white noise value
    // https://www.shadertoy.com/view/4t2SDh
    DoTest("out_gradient_white_triangle_reshape.png", gradient,
        [&rng] (size_t ix, size_t iy)
        {
            static std::uniform_real_distribution<float> dist;
            float nrnd0 = dist(rng);

            // Convert uniform distribution into triangle-shaped distribution.
            float orig = nrnd0 * 2.0f - 1.0f;
            nrnd0 = orig * 1.0f / sqrt(abs(orig));
            nrnd0 = std::max(-1.0f, nrnd0); // Nerf the NaN generated by 0*rsqrt(0). Thanks @FioraAeterna!
            nrnd0 = nrnd0 - sign(orig) + 0.5f;

            // Result is range [-0.5,1.5] which is
            // useful for actual dithering.
            // convert to [0,1] for histogram.
            return (nrnd0 + 0.5f) * 0.5f;
        }
    );
    // TODO: try working through inverting the CDF for triangle distribution

    // TODO: apparently the -0.5 to +1.5 is useful for dithering?? understand that!
    // TODO: test multiple quantizations? not just 1 bit? i think thats where the triangular noise helps maybe

    // TODO: if adding 2 white noise together, does it hurt the DFT? maybe do a DFT of dither pattern and see?

    return 0;
}


/*

TODO:

- blue noise dither a gradient using void and cluster generated noise. Load the texture!
- triangular white/blue noise a gradient (how to get triangular blue noise? maybe just white first?)

- DFT of noise? or is linking to previous post enough
- threshold tests of blue noise along with histogram and DFT?



Notes:
* working in float because we need to dither in linear color space.
* find "inside" that talks about triangular distributed noise
 * page 54 here. there's a link to a paper too. https://www.gdcvault.com/play/1023002/Low-Complexity-High-Fidelity-INSIDE
 * paper: https://uwspace.uwaterloo.ca/bitstream/handle/10012/3867/thesis.pdf;jsessionid=74681FAF2CA22E754C673E9A1E6957EC?sequence=1
* link to last blog post about noise color being independent of distribution?

Future:
! DFT circle by packing hexagons?

*/