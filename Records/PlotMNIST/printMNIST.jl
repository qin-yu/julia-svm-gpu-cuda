using MLDatasets, ImageView
using Gtk, Cairo

function write_to_png(guidict, filename)
    canvas = guidict["gui"]["canvas"]
    ctx = getgc(canvas)
    Cairo.write_to_png(ctx.surface, filename)
end

𝑋_ALL_train, 𝒚_ALL_train = MNIST.traindata(Float32, 1:10)

guidict = imshow(𝑋_ALL_train[:,:,5]')

for i=1:10
    guidict = imshow(𝑋_ALL_train[:,:,i]')
    sleep(3)
    write_to_png(guidict, "$(i)th_MNIST_is_$(𝒚_ALL_train[i]).png")
end
