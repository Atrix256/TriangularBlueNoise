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



struct Image
{
    Image()
    {
    }

    Image(size_t w, size_t h)
    {
        width = w;
        height = h;
        pixels.resize(w*h);
    }

    std::vector<float> pixels;
    size_t width = 0;
    size_t height = 0;
};

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
void AppendImageVertical(Image& dest, const Image& src)
{
    assert(dest.width == src.width);

    size_t destSize = dest.pixels.size();
    size_t srcSize = src.pixels.size();
    dest.pixels.resize(destSize + srcSize);
    memcpy(&dest.pixels[destSize], src.pixels.data(), srcSize * sizeof(src.pixels[0]));

    dest.height += src.height;
}

void SaveImage(const char* fileName, const Image& image)
{
    std::vector<uint8> outImage(image.pixels.size());
    for (size_t index = 0; index < image.pixels.size(); ++index)
        outImage[index] = uint8(Clamp(image.pixels[index] * 255.0f + 0.5f, 0.0f, 255.0f));

    stbi_write_png(fileName, int(image.width), int(image.height), 1, outImage.data(), 0);
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

Image LinearTosRGB(const Image& image)
{
    Image ret = image;
    for (float& f : ret.pixels)
        f = linear_to_srgb_float(f);
    return ret;
}

float Quantize(float value, size_t quantizationLevels)
{
    return Clamp(floor(value * float(quantizationLevels)) / float(quantizationLevels), 0.0f, float(quantizationLevels));
}

template <typename LAMBDA>
void DoTest(const char* baseFileName, const char* name, const Image& srcImage, float randMin, float randMax, bool subtractive, const LAMBDA& lambda)
{
    char fileName[256];
    sprintf(fileName, baseFileName, name);

    Image noise = srcImage;
    Image gradientDithered = srcImage;
    Image gradientAbsError = srcImage;
    Image gradientNormalizedError = srcImage;
    Image gradientHistogram = srcImage;
    std::vector<uint32> histogram(srcImage.width, 0);

    float errorMin = FLT_MAX;
    float errorMax = -FLT_MAX;

    size_t quantizationLevels = 6;

    for (size_t iy = 0; iy < srcImage.height; ++iy)
    {
        for (size_t ix = 0; ix < srcImage.width; ++ix)
        {
            // dither and quantize
            float randValueRaw = lambda(ix, iy);
            float randValue = randValueRaw / float(quantizationLevels);
            float ditheredValue = Quantize(srcImage.pixels[iy*srcImage.width + ix] + randValue, quantizationLevels);

            // subtract the noise after quantization if we are doing subtractive dithering
            if (subtractive)
                ditheredValue -= randValue;

            gradientDithered.pixels[iy*srcImage.width + ix] = ditheredValue;

            // error image
            float error = srcImage.pixels[iy*srcImage.width + ix] - ditheredValue;
            gradientAbsError.pixels[iy*srcImage.width + ix] = std::abs(error);
            gradientNormalizedError.pixels[iy*srcImage.width + ix] = error;
            errorMin = std::min(errorMin, error);
            errorMax = std::max(errorMax, error);

            // histogram
            float randValueHistogram = (randValueRaw - randMin) / (randMax - randMin);
            size_t histogramBucket = Clamp<size_t>(size_t(randValueHistogram * (srcImage.width - 1) + 0.5f), 0, srcImage.width - 1);
            histogram[histogramBucket]++;

            // make noise image
            noise.pixels[iy*srcImage.width + ix] = randValueHistogram;
        }
    }

    // normalize the error
    for (float& f : gradientNormalizedError.pixels)
        f = (f - errorMin) / (errorMax - errorMin);

    // make histogram image
    uint32 maxHistogramValue = histogram[0];
    for (uint32 histogramValue : histogram)
        maxHistogramValue = std::max(maxHistogramValue, histogramValue);

    for (size_t iy = 0; iy < srcImage.height; ++iy)
    {
        for (size_t ix = 0; ix < srcImage.width; ++ix)
        {
            float normalizedHistogramValue = 1.0f - float(histogram[ix]) / float(maxHistogramValue);
            size_t height = Clamp<size_t>(size_t(normalizedHistogramValue * (srcImage.height - 1) + 0.5), 0, srcImage.height - 1);
            gradientHistogram.pixels[iy*srcImage.width + ix] = (iy < height) ? 1.0f : 0.0f;
        }
    }

    Image outImage = srcImage;
    //AppendImageVertical(outImage, noise);
    AppendImageVertical(outImage, gradientDithered);
    //AppendImageVertical(outImage, LinearTosRGB(gradientAbsError));
    AppendImageVertical(outImage, LinearTosRGB(gradientNormalizedError));
    AppendImageVertical(outImage, gradientHistogram);
    SaveImage(fileName, outImage);
}

// https://www.shadertoy.com/view/4t2SDh
float ReshapeUniformToTriangle(float rnd)
{
    float orig = rnd * 2.0f - 1.0f;
    rnd = std::max(-1.0f, orig * 1.0f / sqrt(abs(orig)));
    rnd = rnd - sign(orig) + 0.5f;
    return rnd;
}

void DoTests(const Image& srcImage, const char* name)
{
    // naked quantization tests
    DoTest("out/%s_none.png", name, srcImage, 0.0f, 1.0f, false,
        [](size_t ix, size_t iy)
        {
            return 0.0f;
        }
    );
    DoTest("out/%s_round.png", name, srcImage, 0.0f, 1.0f, false,
        [](size_t ix, size_t iy)
        {
            return 0.5f;
        }
    );

    // uniform white noise test
    DoTest("out/%s_white_1.png", name, srcImage, 0.0f, 1.0f, false,
        [] (size_t ix, size_t iy)
        {
            static std::mt19937 rng(GetRNGSeed());
            static std::uniform_real_distribution<float> dist;
            return dist(rng);
        }
    );

    // triangular white noise test, made by combining two white noise values
    DoTest("out/%s_white_2.png", name, srcImage, -0.5f, 1.5f, false,
        [] (size_t ix, size_t iy)
        {
            static std::mt19937 rng(GetRNGSeed());
            static std::uniform_real_distribution<float> dist;
            return dist(rng) + dist(rng) - 0.5f;
        }
    );

    // triangular white noise test, made by reshaping a single white noise value
    DoTest("out/%s_white_2_reshape.png", name, srcImage, -0.5f, 1.5f, false,
        [] (size_t ix, size_t iy)
        {
            static std::mt19937 rng(GetRNGSeed());
            static std::uniform_real_distribution<float> dist;
            return ReshapeUniformToTriangle(dist(rng));
        }
    );

    // gaussian-ish white noise test, made by combining 4 white noise values
    DoTest("out/%s_white_4.png", name, srcImage, -1.5f, 2.5f, false,
        [] (size_t ix, size_t iy)
        {
            static std::mt19937 rng(GetRNGSeed());
            static std::uniform_real_distribution<float> dist;
            return dist(rng) + dist(rng) + dist(rng) + dist(rng) - 1.5f;
        }
    );

    // gaussian-ish white noise test, made by combining 8 white noise values
    DoTest("out/%s_white_8.png", name, srcImage, -3.5f, 4.5f, false,
        [] (size_t ix, size_t iy)
        {
            static std::mt19937 rng(GetRNGSeed());
            static std::uniform_real_distribution<float> dist;
            return dist(rng) + dist(rng) + dist(rng) + dist(rng) + dist(rng) + dist(rng) + dist(rng) + dist(rng) - 3.5f;
        }
    );

    // gaussian-ish white noise test, made by combining 16 white noise values
    DoTest("out/%s_white_16.png", name, srcImage, -7.5f, 8.5f, false,
        [](size_t ix, size_t iy)
        {
            static std::mt19937 rng(GetRNGSeed());
            static std::uniform_real_distribution<float> dist;
            return dist(rng) + dist(rng) + dist(rng) + dist(rng) + dist(rng) + dist(rng) + dist(rng) + dist(rng) +
                   dist(rng) + dist(rng) + dist(rng) + dist(rng) + dist(rng) + dist(rng) + dist(rng) + dist(rng) -
                   7.5f;
        }
    );

    // subtractive dithering uniform white noise test
    DoTest("out/%s_white_1_subtractive.png", name, srcImage, 0.0f, 1.0f, true,
        [] (size_t ix, size_t iy)
        {
            static std::mt19937 rng(GetRNGSeed());
            static std::uniform_real_distribution<float> dist;
            return dist(rng);
        }
    );

    // blue noise
    {
        int w, h, c;
        uint8* bna = stbi_load("BlueNoise64_A.png", &w, &h, &c, 4);
        uint8* bnb = stbi_load("BlueNoise64_B.png", &w, &h, &c, 4);

        // uniform blue noise test
        DoTest("out/%s_blue_1.png", name, srcImage, 0.0f, 1.0f, false,
            [=] (size_t ix, size_t iy)
            {
                ix = ix % w;
                iy = iy % h;
                return float(bna[(iy*w + ix) * 4]) / 255.0f;
            }
        );

        // triangular blue noise test, made by combining two blue noise values
        DoTest("out/%s_blue_2.png", name, srcImage, -0.5f, 1.5f, false,
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
        DoTest("out/%s_blue_2_reshape.png", name, srcImage, -0.5f, 1.5f, false,
            [=](size_t ix, size_t iy)
            {
                ix = ix % w;
                iy = iy % h;
                float value = float(bna[(iy*w + ix) * 4]) / 255.0f;
                return ReshapeUniformToTriangle(value);
            }
        );

        // subtractive dithering uniform blue noise test
        DoTest("out/%s_blue_1_subtractive.png", name, srcImage, 0.0f, 1.0f, true,
            [=] (size_t ix, size_t iy)
            {
                ix = ix % w;
                iy = iy % h;
                return float(bna[(iy*w + ix) * 4]) / 255.0f;
            }
        );

        stbi_image_free(bna);
        stbi_image_free(bnb);
    }
}

int main(int argc, char** argv)
{
    // do tests on a gradient
    {
        Image gradient(c_gradientWidth, c_gradientHeight);
        {
            for (size_t iy = 0; iy < c_gradientHeight; ++iy)
                for (size_t ix = 0; ix < c_gradientWidth; ++ix)
                    gradient.pixels[iy*c_gradientWidth + ix] = float(ix) / float(c_gradientWidth - 1);
        }

        DoTests(gradient, "gradient");
    }

    // do test on a photo
    {
        int w, h, c;
        uint8* sceneryImg = stbi_load("scenery.png", &w, &h, &c, 4);

        Image scenery(w, h);
        const uint8* ptr = sceneryImg;
        for (size_t i = 0; i < w*h; ++i)
        {
            scenery.pixels[i] = Clamp((float(ptr[0]) * 0.3f + float(ptr[1]) * 0.59f + float(ptr[2]) * 0.11f) / 255.0f, 0.0f, 1.0f);
            ptr += 4;
        }

        DoTests(scenery, "scenery");

        stbi_image_free(sceneryImg);
    }

    // TODO: if adding 2 blue noise together, does it hurt the DFT? maybe do a DFT of dither pattern and see?
    // It looks like it does which makes sense. it's introducing white noise. could maybe try void and cluster to make float values, and then reshape?
    // could also try loading an HDR file from the free blue noise texture site

    return 0;
}


/*

TODO:

* do subtractive dither
* maybe show mean and variance (first 2 moments?)

* only show normalized error by default, not abs error

* do sRGB examples too: linear to srgb, dither, back to linear?

* triangle dithering at boundaries, uniform in the middle, like mikkel talks about

! subtractive dithering has smaller noise magnitude!

! confusing: they should floyd steinberg dithering which made blue noise (ish? or actual?) results. They showed how adding dithering (white noise) improved it. page 56


Blog:
* show abs error vs normalized error.  Normalized error shows the actual error pattern, regardless of signal.
"HINDSIGHT: this shows the absolute value of the error, abs(err), which makes it hard
to see that positive/negative errors “cancel each out” in the noisy areas. Remapping
the actual error to a [0;1] range gives a much more accurate depiction of the error."

* show differences between abs error and normalized error

Mention that this might be good for textures on disk!
 * also for gbuffers and similar 

? should you show multiple quantilzation levels? it might help with triangle dithering at boundaries, to have a low bucket count.

! i do think that when low discrepancy animating noise over time, that you should be using triangular distributed noise. 1 bit vs N bit.

Notes:
* working in float because we need to dither in linear color space.
* find "inside" that talks about triangular distributed noise
 * this http://loopit.dk/banding_in_games.pdf#page=33
 * page 54 here. there's a link to a paper too. https://www.gdcvault.com/play/1023002/Low-Complexity-High-Fidelity-INSIDE
 * paper: https://uwspace.uwaterloo.ca/bitstream/handle/10012/3867/thesis.pdf;jsessionid=74681FAF2CA22E754C673E9A1E6957EC?sequence=1
 * also this: http://gpuopen.com/wp-content/uploads/2016/03/GdcVdrLottes.pdf#page=83
* mention this: "GPUs round, so dither-range should be [-1;1[". not -0.5 to 1.5

* link to last blog post about noise color being independent of distribution?
* "The error resulting from a triangularly distributed noise is independent of the signal."
* when quantizing for eg 3 levels, you can quantize to 0/3, 1/3, 2/3.  OR can quantize to 0/2, 1/2, 2/2.  This is doing the first way because it's better for dithering due to last bucket. As bucket count goes up, the choice matters less.
* sRGB and dithering: https://twitter.com/Atrix256/status/1179971512461225984?s=20
* Mikkel mentioned that -1 to +1 is better on gpus due to rounding


Why TPDF?  it controls the second moment. white noise only controls the first.
TPDF dither does not control the third or higher moments of the total error signal. However, after extensive testing on the subject, it is generally agreed that the human ear is not
sensitive to statistical moments higher than the second. This means that trying to control total
error moments higher than the second moment is unnecessary for audio applications, but could
be relevant in some measurement applications.



Future:
! DFT circle by packing hexagons?
* subtractive dither? (seems neat! subtract dither on playback side)
* denoising stuff - guided filter

*/
