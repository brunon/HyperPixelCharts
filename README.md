# HyperPixelCharts

RaspberryPi project using a [HyperPixel 4.0 display](https://shop.pimoroni.com/products/hyperpixel-4?variant=12569539706963) to render pre-generated image files / charts

This project includes the TCP socket-based server process which accepts the following network commands:
- `on` : turn LCD on
- `off` : turn LCD off
- `previous` : show the previous image from the configured directory
- `next` : show the next image from the configured directory

The server will display the image in full screen on the HyperPixel display, and will monitor the image file for changes (via polling) and automatically refresh the image shown if the file changes.

This uses a low-level X11 utility called `feh` to display the image full-screen (install with `sudo apt-get install feh`)
