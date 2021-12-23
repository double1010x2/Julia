#!/usr/bin/env julia

# Calculate Julia set in Julia

import GR

function julia(z, c, iters)
    ci = 0
    inc = 1

    for i in 0:iters
        z = z^2 + c
        if abs2(z) >= 4
            return ci
        end
        ci += inc
        if ci == 0 || ci == 255
            inc = -inc
        end
    end

    return 255
end

function main()

    function create_fractal(min_x, max_x, min_y, max_y, image, iters)
        height = size(image, 1)
        width = size(image, 2)

        pixel_size_x = (max_x - min_x) / width
        pixel_size_y = (max_y - min_y) / height
        for i in 1:width
            real = min_x + (i - 1) * pixel_size_x
            for j in 1:height
                imag = min_y + (j - 1) * pixel_size_y
                color = julia(complex(real, imag), seed, iters)
                image[j, i] = color
            end
        end
    end

    seed = complex(-0.156844471694257101941, -0.649707745759247905171)
    x, y = (-0.16, -0.64)

    f = 1.5
    for i in 0:100
        image = zeros(Int32, 500, 500)

        dt = @elapsed create_fractal(x-f, x+f, y-f, y+f, image, 500)
        println("Julia set created in $dt s")

        GR.clearws()
        GR.setviewport(0, 1, 0, 1)
        GR.setcolormap(13)
        GR.cellarray(0, 1, 0, 1, 500, 500, image .+ 1000)
        GR.updatews()

        f *= 0.9
    end
end

main()
