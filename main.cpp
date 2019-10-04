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

static const size_t c_gradientWidth = 512;
static const size_t c_gradientHeight = 64;



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
        outImage[index] = uint8(Clamp(image[index] * 255.0f + 0.5f, 0.0f, 255.0f));

    stbi_write_png(fileName, int(width), int(height), 1, outImage.data(), 0);
}

std::vector<float> sRGBToLinear(const std::vector<float>& image)
{
    std::vector<float> ret = image;
    for (float& f : ret)
        f = srgb_to_linear_float(f);
    return ret;
}

std::vector<float> LinearTosRGB(const std::vector<float>& image)
{
    std::vector<float> ret = image;
    for (float& f : ret)
        f = linear_to_srgb_float(f);
    return ret;
}

float Quantize(float value, size_t quantizationLevels)
{
    return Clamp(floor(value * float(quantizationLevels)) / float(quantizationLevels), 0.0f, float(quantizationLevels));
}

template <typename LAMBDA>
void DoTest(const char* fileName, std::vector<float> gradient, float randMin, float randMax, bool sRGBBeforeDither, const LAMBDA& lambda)
{
    std::vector<float> noise(c_gradientWidth*c_gradientHeight);
    std::vector<float> gradientDithered(c_gradientWidth*c_gradientHeight);
    std::vector<float> gradientError(c_gradientWidth*c_gradientHeight);
    std::vector<float> gradientHistogram(c_gradientWidth*c_gradientHeight);
    std::vector<uint32> histogram(c_gradientWidth, 0);

    size_t quantizationLevels = 6;

    if (sRGBBeforeDither)
        gradient = LinearTosRGB(gradient);

    for (size_t iy = 0; iy < c_gradientHeight; ++iy)
    {
        for (size_t ix = 0; ix < c_gradientWidth; ++ix)
        {
            // dither and quantize
            float randValueRaw = lambda(ix, iy);
            float randValue = randValueRaw / float(quantizationLevels);
            float ditheredValue = Quantize(gradient[iy*c_gradientWidth + ix] + randValue, quantizationLevels);
            gradientDithered[iy*c_gradientWidth + ix] = ditheredValue;

            // error image
            gradientError[iy*c_gradientWidth + ix] = std::abs(gradient[iy*c_gradientWidth + ix] - ditheredValue);

            // histogram
            float randValueHistogram = (randValueRaw - randMin) / (randMax - randMin);
            size_t histogramBucket = Clamp<size_t>(size_t(randValueHistogram * (c_gradientWidth - 1) + 0.5f), 0, c_gradientWidth - 1);
            histogram[histogramBucket]++;

            // make noise image
            noise[iy*c_gradientWidth + ix] = randValueHistogram;
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

    if (!sRGBBeforeDither)
    {
        gradient = LinearTosRGB(gradient);
    }

    // TODO: which of these images needs to be converted from linear to sRGB? They may be conditional too. think about em!

    std::vector<float> outImage = gradient;
    //AppendImageVertical(outImage, noise);
    AppendImageVertical(outImage, gradientDithered);
    AppendImageVertical(outImage, LinearTosRGB(gradientError));
    AppendImageVertical(outImage, LinearTosRGB(gradientHistogram));
    SaveImage(fileName, outImage, c_gradientWidth, c_gradientHeight * 4);
}

// https://www.shadertoy.com/view/4t2SDh
float ReshapeUniformToTriangle(float rnd)
{
    float orig = rnd * 2.0f - 1.0f;
    rnd = std::max(-1.0f, orig * 1.0f / sqrt(abs(orig)));
    rnd = rnd - sign(orig) + 0.5f;
    return rnd;
}

int main(int argc, char** argv)
{
    // make the (linear) gradient image
    std::vector<float> gradient(c_gradientWidth*c_gradientHeight);
    {
        for (size_t iy = 0; iy < c_gradientHeight; ++iy)
            for (size_t ix = 0; ix < c_gradientWidth; ++ix)
                gradient[iy*c_gradientWidth + ix] = float(ix) / float(c_gradientWidth - 1);
    }

    // naked quantization tests
    DoTest("out_none.png", gradient, 0.0f, 1.0f, true,
        [](size_t ix, size_t iy)
        {
            return 0.0f;
        }
    );
    DoTest("out_round.png", gradient, 0.0f, 1.0f, true,
        [](size_t ix, size_t iy)
        {
            return 0.5f;
        }
    );

    // uniform white noise test
    DoTest("out_white_1.png", gradient, 0.0f, 1.0f, true,
        [] (size_t ix, size_t iy)
        {
            static std::mt19937 rng(GetRNGSeed());
            static std::uniform_real_distribution<float> dist;
            return dist(rng);
        }
    );

    // triangular white noise test, made by combining two white noise values
    DoTest("out_white_2.png", gradient, -0.5f, 1.5f, true,
        [] (size_t ix, size_t iy)
        {
            static std::mt19937 rng(GetRNGSeed());
            static std::uniform_real_distribution<float> dist;
            return dist(rng) + dist(rng) - 0.5f;
        }
    );

    // triangular white noise test, made by reshaping a single white noise value
    DoTest("out_white_2_reshape.png", gradient, -0.5f, 1.5f, true,
        [] (size_t ix, size_t iy)
        {
            static std::mt19937 rng(GetRNGSeed());
            static std::uniform_real_distribution<float> dist;
            return ReshapeUniformToTriangle(dist(rng));
        }
    );

    // triangular white noise test, made by reshaping a single white noise value
    // Dithering before sRGB (bad!)
    DoTest("out_white_2_reshape_linear.png", gradient, -0.5f, 1.5f, false,
        [](size_t ix, size_t iy)
        {
            static std::mt19937 rng(GetRNGSeed());
            static std::uniform_real_distribution<float> dist;
            return ReshapeUniformToTriangle(dist(rng));
        }
    );

    // gaussian-ish white noise test, made by combining 4 white noise values
    DoTest("out_white_4.png", gradient, -1.5f, 2.5f, true,
        [] (size_t ix, size_t iy)
        {
            static std::mt19937 rng(GetRNGSeed());
            static std::uniform_real_distribution<float> dist;
            return dist(rng) + dist(rng) + dist(rng) + dist(rng) - 1.5f;
        }
    );

    // gaussian-ish white noise test, made by combining 8 white noise values
    DoTest("out_white_8.png", gradient, -3.5f, 4.5f, true,
        [] (size_t ix, size_t iy)
        {
            static std::mt19937 rng(GetRNGSeed());
            static std::uniform_real_distribution<float> dist;
            return dist(rng) + dist(rng) + dist(rng) + dist(rng) + dist(rng) + dist(rng) + dist(rng) + dist(rng) - 3.5f;
        }
    );

    // gaussian-ish white noise test, made by combining 16 white noise values
    DoTest("out_white_16.png", gradient, -7.5f, 8.5f, true,
        [](size_t ix, size_t iy)
    {
        static std::mt19937 rng(GetRNGSeed());
        static std::uniform_real_distribution<float> dist;
        return dist(rng) + dist(rng) + dist(rng) + dist(rng) + dist(rng) + dist(rng) + dist(rng) + dist(rng) +
               dist(rng) + dist(rng) + dist(rng) + dist(rng) + dist(rng) + dist(rng) + dist(rng) + dist(rng) -
               7.5f;
    }
    );

    // blue noise
    {
        int w, h, c;
        uint8* bna = stbi_load("BlueNoise64_A.png", &w, &h, &c, 4);
        uint8* bnb = stbi_load("BlueNoise64_B.png", &w, &h, &c, 4);

        // uniform blue noise test
        DoTest("out_blue_1.png", gradient, 0.0f, 1.0f, true,
            [=] (size_t ix, size_t iy)
            {
                ix = ix % w;
                iy = iy % h;
                return float(bna[(iy*w + ix) * 4]) / 255.0f;
            }
        );

        // triangular blue noise test, made by combining two blue noise values
        DoTest("out_blue_2.png", gradient, -0.5f, 1.5f, true,
            [=] (size_t ix, size_t iy)
            {
                ix = ix % w;
                iy = iy % h;
                float valueA = float(bna[(iy*w + ix) * 4]) / 255.0f;
                float valueB = float(bnb[(iy*w + ix) * 4]) / 255.0f;
                return valueA + valueB - 0.5f;
            }
        );

        // triangular blue noise test, made by reshaping a single blue noise value
        DoTest("out_blue_2_reshape.png", gradient, -0.5f, 1.5f, true,
            [=](size_t ix, size_t iy)
            {
                ix = ix % w;
                iy = iy % h;
                float value = float(bna[(iy*w + ix) * 4]) / 255.0f;
                return ReshapeUniformToTriangle(value);
            }
        );

        // triangular blue noise test, made by reshaping a single blue noise value
        // Dithering before sRGB (bad!)
        DoTest("out_blue_2_reshape_linear.png", gradient, -0.5f, 1.5f, false,
            [=](size_t ix, size_t iy)
            {
                ix = ix % w;
                iy = iy % h;
                float value = float(bna[(iy*w + ix) * 4]) / 255.0f;
                return ReshapeUniformToTriangle(value);
            }
        );

        // TODO: why is reshaped blue noise so bad? maybe cause it isn't a float coming in, but a uint8?
        // TODO: blue noise by averaging is ugly ):

        stbi_image_free(bna);
        stbi_image_free(bnb);
    }

    // TODO: try working through inverting the CDF for triangle distribution

    // TODO: apparently the -0.5 to +1.5 is useful for dithering?? understand that!
    // TODO: test multiple quantizations? not just 1 bit? i think thats where the triangular noise helps maybe

    // TODO: if adding 2 blue noise together, does it hurt the DFT? maybe do a DFT of dither pattern and see?
    // It looks like it does which makes sense. it's introducing white noise. could maybe try void and cluster to make float values, and then reshape?
    // could also try loading an HDR file from the free blue noise texture site

    return 0;
}


/*

TODO:

The problem is you need to dither AFTER sRGB. Show the difference to show how bad it is!
 * get sRGB out of saving images.
 * convert them on an as needed basis
 * figure out which should / should not be converted!

- DFT of noise? or is linking to previous post enough. actually it would be nice to see frequencies i guess.
- threshold tests of blue noise along with histogram and DFT?



Notes:
* working in float because we need to dither in linear color space.
* find "inside" that talks about triangular distributed noise
 * page 54 here. there's a link to a paper too. https://www.gdcvault.com/play/1023002/Low-Complexity-High-Fidelity-INSIDE
 * paper: https://uwspace.uwaterloo.ca/bitstream/handle/10012/3867/thesis.pdf;jsessionid=74681FAF2CA22E754C673E9A1E6957EC?sequence=1
* link to last blog post about noise color being independent of distribution?
* "The error resulting from a triangularly distributed noise is independent of the signal."
* when quantizing for eg 3 levels, you can quantize to 0/3, 1/3, 2/3.  OR can quantize to 0/2, 1/2, 2/2.  This is doing the first way because it's better for dithering due to last bucket. As bucket count goes up, the choice matters less.
* sRGB and dithering: https://twitter.com/Atrix256/status/1179971512461225984?s=20

Future:
! DFT circle by packing hexagons?

*/