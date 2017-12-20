import matplotlib.image as img
import numpy as np

def make_gif(images, fname, duration=2, true_image=False):
  import moviepy.editor as mpy

  def make_frame(t):
    try:
      x = images[int(len(images)/duration*t)]
    except:
      x = images[-1]

    if true_image:
      return x.astype(np.uint8)
    else:
      return (x*255).astype(np.uint8)

  clip = mpy.VideoClip(make_frame, duration=duration)
  clip.write_gif(fname, fps = len(images) / duration)

mode = 'gan'
path_template = 'smpl_mnist/256_{}/train_{:04d}.png'

images = []
for i in xrange(100):
    images.append(img.imread(path_template.format(mode, i)))

print(images[-1])
print(images[-1].size)
print(np.amax(images[-1]))
print(np.amin(images[-1]))

make_gif(images, '{}.gif'.format(mode), duration=3, true_image=False)
