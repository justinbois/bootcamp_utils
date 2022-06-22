import warnings

import numpy as np
import skimage

import bokeh.models
import bokeh.palettes
import bokeh.plotting


def imshow(
    im,
    color_mapper=None,
    frame_height=400,
    frame_width=None,
    length_units="pixels",
    interpixel_distance=1.0,
    x_range=None,
    y_range=None,
    colorbar=False,
    no_ticks=False,
    x_axis_label=None,
    y_axis_label=None,
    title=None,
    flip=True,
    return_im=False,
    saturate_channels=True,
    min_intensity=None,
    max_intensity=None,
    display_clicks=False,
):
    """
    Display an image in a Bokeh figure.

    Parameters
    ----------
    im : Numpy array
        If 2D, intensity image to be displayed. If 3D, first two
        dimensions are pixel values. Last dimension can be of length
        1, 2, or 3, which specify colors.
    color_mapper : str or bokeh.models.LinearColorMapper, default None
        If `im` is an intensity image, `color_mapper` is a mapping of
        intensity to color. If None, default is 256-level Viridis.
        If `im` is a color image, then `color_mapper` can either be
        'rgb' or 'cmy' (default), for RGB or CMY merge of channels.
    frame_height : int
        Height of the plot in pixels. The width is scaled so that the
        x and y distance between pixels is the same.
    frame_width : int or None (default)
        If None, the width is scaled so that the x and y distance
        between pixels is approximately the same. Otherwise, the width
        of the plot in pixels.
    length_units : str, default 'pixels'
        The units of length in the image.
    interpixel_distance : float, default 1.0
        Interpixel distance in units of `length_units`.
    x_range : bokeh.models.Range1d instance, default None
        Range of x-axis. If None, determined automatically.
    y_range : bokeh.models.Range1d instance, default None
        Range of y-axis. If None, determined automatically.
    colorbar : bool, default False
        If True, include a colorbar.
    no_ticks : bool, default False
        If True, no ticks are displayed. See note below.
    x_axis_label : str, default None
        Label for the x-axis. If None, labeled with `length_units`.
    y_axis_label : str, default None
        Label for the y-axis. If None, labeled with `length_units`.
    title : str, default None
        The title of the plot.
    flip : bool, default True
        If True, flip image so it displays right-side up. This is
        necessary because traditionally images have their 0,0 pixel
        index in the top left corner, and not the bottom left corner.
    return_im : bool, default False
        If True, return the GlyphRenderer instance of the image being
        displayed.
    saturate_channels : bool, default True
        If True, each of the channels have their displayed pixel values
        extended to range from 0 to 255 to show the full dynamic range.
    min_intensity : int or float, default None
        Minimum possible intensity of a pixel in the image. If None,
        the image is scaled based on the dynamic range in the image.
    max_intensity : int or float, default None
        Maximum possible intensity of a pixel in the image. If None,
        the image is scaled based on the dynamic range in the image.
    display_clicks : bool, default False
        If True, display clicks to the right of the plot using
        JavaScript. The clicks are not recorded nor stored, just
        printed.

    Returns
    -------
    p : bokeh.plotting.figure instance
        Bokeh plot with image displayed.
    im : bokeh.models.renderers.GlyphRenderer instance (optional)
        The GlyphRenderer instance of the image being displayed. This is
        only returned if `return_im` is True.
    """

    # If a single channel in 3D image, flatten and check shape
    if im.ndim == 3:
        if im.shape[2] == 1:
            im = im[:, :, 0]
        elif im.shape[2] not in [2, 3]:
            raise RuntimeError("Can only display 1, 2, or 3 channels.")

    # If binary image, make sure it's int
    if im.dtype == bool:
        im = im.astype(np.uint8)

    # If it's above 8-bit, convert to float
    if im.max() > 255:
        im = im.astype(float) / im.max()

    # Get color mapper
    if im.ndim == 2:
        if color_mapper is None:
            color_mapper = bokeh.models.LinearColorMapper(bokeh.palettes.viridis(256))
        elif type(color_mapper) == str and color_mapper.lower() in ["rgb", "cmy"]:
            raise RuntimeError("Cannot use rgb or cmy colormap for intensity image.")
        if min_intensity is None:
            color_mapper.low = im.min()
        else:
            color_mapper.low = min_intensity
        if max_intensity is None:
            color_mapper.high = im.max()
        else:
            color_mapper.high = max_intensity
    elif im.ndim == 3:
        if color_mapper is None or color_mapper.lower() == "cmy":
            im = im_merge(
                *np.rollaxis(im, 2),
                cmy=True,
                im_0_min=min_intensity,
                im_1_min=min_intensity,
                im_2_min=min_intensity,
                im_0_max=max_intensity,
                im_1_max=max_intensity,
                im_2_max=max_intensity,
            )
        elif color_mapper.lower() == "rgb":
            im = im_merge(
                *np.rollaxis(im, 2),
                cmy=False,
                im_0_min=min_intensity,
                im_1_min=min_intensity,
                im_2_min=min_intensity,
                im_0_max=max_intensity,
                im_1_max=max_intensity,
                im_2_max=max_intensity,
            )
        else:
            raise RuntimeError("Invalid color mapper for color image.")
    else:
        raise RuntimeError("Input image array must have either 2 or 3 dimensions.")

    # Get shape, dimensions
    n, m = im.shape[:2]
    if x_range is not None and y_range is not None:
        dw = x_range[1] - x_range[0]
        dh = y_range[1] - y_range[0]
    else:
        dw = m * interpixel_distance
        dh = n * interpixel_distance
        x_range = [0, dw]
        y_range = [0, dh]

    # Set up figure with appropriate dimensions
    if frame_width is None:
        frame_width = int(m / n * frame_height)
    if colorbar:
        toolbar_location = "above"
    else:
        toolbar_location = "right"
    p = bokeh.plotting.figure(
        frame_height=frame_height,
        frame_width=frame_width,
        x_range=x_range,
        y_range=y_range,
        title=title,
        toolbar_location=toolbar_location,
        tools="pan,box_zoom,wheel_zoom,save,reset",
    )

    if no_ticks:
        p.xaxis.major_label_text_font_size = "0pt"
        p.yaxis.major_label_text_font_size = "0pt"
        p.xaxis.major_tick_line_color = None
        p.xaxis.minor_tick_line_color = None
        p.yaxis.major_tick_line_color = None
        p.yaxis.minor_tick_line_color = None
    else:
        if x_axis_label is None:
            p.xaxis.axis_label = length_units
        else:
            p.xaxis.axis_label = x_axis_label
        if y_axis_label is None:
            p.yaxis.axis_label = length_units
        else:
            p.yaxis.axis_label = y_axis_label

    # Display the image
    if im.ndim == 2:
        if flip:
            im = im[::-1, :]
        im_bokeh = p.image(
            image=[im],
            x=x_range[0],
            y=y_range[0],
            dw=dw,
            dh=dh,
            color_mapper=color_mapper,
        )
    else:
        im_bokeh = p.image_rgba(
            image=[rgb_to_rgba32(im, flip=flip)],
            x=x_range[0],
            y=y_range[0],
            dw=dw,
            dh=dh,
        )

    # Make a colorbar
    if colorbar:
        if im.ndim == 3:
            warnings.warn("No colorbar display for RGB images.")
        else:
            color_bar = bokeh.models.ColorBar(
                color_mapper=color_mapper,
                label_standoff=12,
                border_line_color=None,
                location=(0, 0),
            )
            p.add_layout(color_bar, "right")

    if display_clicks:
        div = bokeh.models.Div(width=200)
        layout = bokeh.layouts.row(p, div)
        p.js_on_event(bokeh.events.Tap, _display_clicks(div, attributes=["x", "y"]))
        if return_im:
            return layout, im_bokeh
        else:
            return layout

    if return_im:
        return p, im_bokeh
    return p


def im_merge(
    im_0,
    im_1,
    im_2=None,
    im_0_max=None,
    im_1_max=None,
    im_2_max=None,
    im_0_min=None,
    im_1_min=None,
    im_2_min=None,
    cmy=True,
):
    """
    Merge channels to make RGB image.

    Parameters
    ----------
    im_0: array_like
        Image represented in first channel.  Must be same shape
        as `im_1` and `im_2` (if not None).
    im_1: array_like
        Image represented in second channel.  Must be same shape
        as `im_1` and `im_2` (if not None).
    im_2: array_like, default None
        Image represented in third channel.  If not None, must be same
        shape as `im_0` and `im_1`.
    im_0_max : float, default max of inputed first channel
        Maximum value to use when scaling the first channel. If None,
        scaled to span entire range.
    im_1_max : float, default max of inputed second channel
        Maximum value to use when scaling the second channel
    im_2_max : float, default max of inputed third channel
        Maximum value to use when scaling the third channel
    im_0_min : float, default min of inputed first channel
        Maximum value to use when scaling the first channel
    im_1_min : float, default min of inputed second channel
        Minimum value to use when scaling the second channel
    im_2_min : float, default min of inputed third channel
        Minimum value to use when scaling the third channel
    cmy : bool, default True
        If True, first channel is cyan, second is magenta, and third is
        yellow. Otherwise, first channel is red, second is green, and
        third is blue.

    Returns
    -------
    output : array_like, dtype float, shape (*im_0.shape, 3)
        RGB image.
    """

    # Compute max intensities if needed
    if im_0_max is None:
        im_0_max = im_0.max()
    if im_1_max is None:
        im_1_max = im_1.max()
    if im_2 is not None and im_2_max is None:
        im_2_max = im_2.max()

    # Compute min intensities if needed
    if im_0_min is None:
        im_0_min = im_0.min()
    if im_1_min is None:
        im_1_min = im_1.min()
    if im_2 is not None and im_2_min is None:
        im_2_min = im_2.min()

    # Make sure maxes are ok
    if (
        im_0_max < im_0.max()
        or im_1_max < im_1.max()
        or (im_2 is not None and im_2_max < im_2.max())
    ):
        raise RuntimeError("Inputted max of channel < max of inputted channel.")

    # Make sure mins are ok
    if (
        im_0_min > im_0.min()
        or im_1_min > im_1.min()
        or (im_2 is not None and im_2_min > im_2.min())
    ):
        raise RuntimeError("Inputted min of channel > min of inputted channel.")

    # Scale the images
    if im_0_max > im_0_min:
        im_0 = (im_0 - im_0_min) / (im_0_max - im_0_min)
    else:
        im_0 = (im_0 > 0).astype(float)

    if im_1_max > im_1_min:
        im_1 = (im_1 - im_1_min) / (im_1_max - im_1_min)
    else:
        im_0 = (im_0 > 0).astype(float)

    if im_2 is None:
        im_2 = np.zeros_like(im_0)
    elif im_2_max > im_2_min:
        im_2 = (im_2 - im_2_min) / (im_2_max - im_2_min)
    else:
        im_0 = (im_0 > 0).astype(float)

    # Convert images to RGB
    if cmy:
        im_c = np.stack((np.zeros_like(im_0), im_0, im_0), axis=2)
        im_m = np.stack((im_1, np.zeros_like(im_1), im_1), axis=2)
        im_y = np.stack((im_2, im_2, np.zeros_like(im_2)), axis=2)
        im_rgb = im_c + im_m + im_y
        for i in [0, 1, 2]:
            im_rgb[:, :, i] /= im_rgb[:, :, i].max()
    else:
        im_rgb = np.empty((*im_0.shape, 3))
        im_rgb[:, :, 0] = im_0
        im_rgb[:, :, 1] = im_1
        im_rgb[:, :, 2] = im_2

    return im_rgb


def rgb_to_rgba32(im, flip=True):
    """
    Convert an RGB image to a 32 bit-encoded RGBA image.

    Parameters
    ----------
    im : ndarray, shape (nrows, ncolums, 3)
        Input image. All pixel values must be between 0 and 1.
    flip : bool, default True
        If True, flip image so it displays right-side up. This is
        necessary because traditionally images have their 0,0 pixel
        index in the top left corner, and not the bottom left corner.

    Returns
    -------
    output : ndarray, shape (nros, ncolumns), dtype np.uint32
        Image decoded as a 32 bit RBGA image.
    """
    # Ensure it has three channels
    if im.ndim != 3 or im.shape[2] != 3:
        raise RuntimeError("Input image is not RGB.")

    # Make sure all entries between zero and one
    if (im < 0).any() or (im > 1).any():
        raise RuntimeError("All pixel values must be between 0 and 1.")

    # Get image shape
    n, m, _ = im.shape

    # Convert to 8-bit, which is expected for viewing
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        im_8 = skimage.img_as_ubyte(im)

    # Add the alpha channel, which is expected by Bokeh
    im_rgba = np.stack(
        (*np.rollaxis(im_8, 2), 255 * np.ones((n, m), dtype=np.uint8)), axis=2
    )

    # Reshape into 32 bit. Must flip up/down for proper orientation
    if flip:
        return np.flipud(im_rgba.view(dtype=np.int32).reshape((n, m)))
    else:
        return im_rgba.view(dtype=np.int32).reshape((n, m))


def rgb_frac_to_hex(rgb_frac):
    """
    Convert fractional RGB values to hexidecimal color string.

    Parameters
    ----------
    rgb_frac : array_like, shape (3,)
        Fractional RGB values; each entry is between 0 and 1.

    Returns
    -------
    str
        Hexidecimal string for the given RGB color.

    Examples
    --------
    >>> rgb_frac_to_hex((0.65, 0.23, 1.0))
    '#a53aff'

    >>> rgb_frac_to_hex((1.0, 1.0, 1.0))
    '#ffffff'
    """

    if len(rgb_frac) != 3:
        raise RuntimeError("`rgb_frac` must have exactly three entries.")

    if (np.array(rgb_frac) < 0).any() or (np.array(rgb_frac) > 1).any():
        raise RuntimeError("RGB values must be between 0 and 1.")

    return "#{0:02x}{1:02x}{2:02x}".format(
        int(rgb_frac[0] * 255), int(rgb_frac[1] * 255), int(rgb_frac[2] * 255)
    )


def _display_clicks(div, attributes=[], style="float:left;clear:left;font_size=0.5pt"):
    """Build a suitable CustomJS to display the current event
    in the div model."""
    return bokeh.models.CustomJS(
        args=dict(div=div),
        code="""
        var attrs = %s; var args = [];
        for (var i=0; i<attrs.length; i++ ) {
            args.push(Number(cb_obj[attrs[i]]).toFixed(4));
        }
        var line = "<span style=%r>[" + args.join(", ") + "], </span>\\n";
        var text = div.text.concat(line);
        var lines = text.split("\\n")
        if ( lines.length > 35 ) { lines.shift(); }
        div.text = lines.join("\\n");
    """
        % (attributes, style),
    )
