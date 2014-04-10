"""

@package        neuron_viewer

@date           Created on Thur April 3, 2014 at 20:20

@author         John Kirkham

"""


import matplotlib as mpl
import matplotlib.figure


from matplotlib.widgets import Slider, Button



class NeuronViewer(matplotlib.figure.Figure):
    """
        @brief      Provides a way to interact with numpy arrays pulled from neuron images.

        @details    Wraps a Matplotlib figure instance.
    """
    def __init__(self, fig, images):
        """
            @brief      Initializes a NeuronViewer using the given figure to clone and image stack to view.

            @details    After cloning, self will be the same as fig. Additional features will be attached to self.

            @param      self    that which is being initialized.
            @param      fig     should be a figure that has been initialized already.
            @param      images  should be a 3d Numpy array

            @todo       Extract constants as parameters of constructor.
        """

        self.__dict__.update(fig.__dict__)
        #fig = None                              # Works. Should it be used?
        self.images = images


        self.subplots_adjust(left=0.25, bottom=0.25)
        self.viewer = self.add_axes([0.25, 0.25, 0.7, 0.7])
        #self.viewer.hold(False)

        self.image_view = self.viewer.imshow(images[0], cmap = mpl.cm.RdBu, vmin = images.min(), vmax = images.max())

        self.colorbar(self.image_view, ax = self.viewer)
        #self.colorbar(image_view, cax = plt.axes([0.1, 0.25, 0.03, 0.65]))

        self.time_nav = TimeNavigator(self, len(images) - 1)

        self.time_nav_cid = self.time_nav.on_time_update(self.time_update)

    def time_update(self):
        """
            @brief  Method to be called by the TimeNavigator when the time changes. Updates image displayed.
        """

        self.image_view.set_array(self.images[self.time_nav.stime.val])
        self.canvas.draw_idle()




class TimeNavigator:
    def __init__(self, fig, max_time, min_time = 0, time_step = 1, axcolor = 'lightgoldenrodyellow', hovercolor = '0.975'):
        """
            @brief      Initializes a TimeNavigator using the given figure and a fixed number of steps.

            @details    Provides a simple slider bar and buttons to navigate through time.

            @param      self        that which is being initialized.
            @param      fig         should be a figure that has been initialized already.
            @param      max_time    maximum step for images.
            @param      min_time    minimum step for images.
            @param      time_step   how much to increase/decrease by for each step.
            @param      axcolor     color to be used for all buttons and slider bar.
            @param      hovercolor  color turned when mouse over occurs for any button.

            @todo       Extract constants as parameters of constructor. Also, determine out a way to make all position relative to some bounding box for all of TimeNavigator as opposed to the fig.
        """

        self.min_time = min_time
        self.max_time = max_time
        self.time_step = time_step
        self.axcolor = axcolor
        self.hovercolor = hovercolor

        self.next_cid = 0

        self.axtime = fig.add_axes([0.25, 0.1, 0.65, 0.03], axisbg = self.axcolor)
        self.stime = Slider(self.axtime, 'Time', self.min_time, self.max_time, valinit = self.min_time, valfmt = '%i')

        self.stime.on_changed(self.time_update)

        self.beginax = fig.add_axes([0.2, 0.025, 0.1, 0.04])
        self.begin_button = Button(self.beginax, 'Begin', color = self.axcolor, hovercolor = self.hovercolor)
        self.begin_button.on_clicked(self.begin_time)

        self.prevax = fig.add_axes([0.3, 0.025, 0.1, 0.04])
        self.prev_button = Button(self.prevax, 'Prev', color = self.axcolor, hovercolor = self.hovercolor)
        self.prev_button.on_clicked(self.prev_time)

        self.nextax = fig.add_axes([0.7, 0.025, 0.1, 0.04])
        self.next_button = Button(self.nextax, 'Next', color = self.axcolor, hovercolor = self.hovercolor)
        self.next_button.on_clicked(self.next_time)

        self.endax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
        self.end_button = Button(self.endax, 'End', color = self.axcolor, hovercolor = self.hovercolor)
        self.end_button.on_clicked(self.end_time)

        self.callbacks = {}

    def begin_time(self, event):
        """
            @brief Sets time to min_time
        """

        #print >> sys.stderr, self.val
        self.time_update(self.min_time)
        #print >> sys.stderr, self.val

    def prev_time(self, event):
        """
            @brief Sets time to one time_step prior
        """

        #print >> sys.stderr, self.val
        self.time_update(self.stime.val - self.time_step)
        #print >> sys.stderr, self.val

    def next_time(self, event):
        """
            @brief Sets time to one time_step after
        """

        #print >> sys.stderr, self.val
        self.time_update(self.stime.val + self.time_step)
        #print >> sys.stderr, self.val

    def end_time(self, event):
        """
            @brief Sets time to max_time
        """

        #print >> sys.stderr, self.val
        self.time_update(self.max_time)
        #print >> sys.stderr, self.val


    def normalize_val(self, val):
        """
            @brief  Takes the time value and normalizes it to fit within the range. Then makes sure it is a discrete number of steps from the min_time.

            @param  self    this instance
            @param  val     float position from the slider bar to correct

            @return the normalized value.
        """

        if val < self.min_time:
            return(self.min_time)
        elif val > self.max_time:
            return(self.max_time)
        else:
            return(int(round((val - self.min_time)/self.time_step)))


    def time_update(self, val):
        """
            @brief  Takes the time value and normalizes it within the range if it does not fit.

            @param  self    this instance
            @param  val     float position from slider bar to move to
        """

        val = self.normalize_val(val)

        if val != self.stime.val:
            self.stime.set_val(val)

            for each_cid, each_callback in self.callbacks.items():
                #print >> sys.stderr, "Called ", each_cid
                each_callback()

            return

    def disconnect(self, cid):
        """
            @brief  Disconnects the given cid from being notified of time updates.

            @param  self    this instance
            @param  cid     ID of callback to pull
        """

        del self.callbacks[cid]

    def on_time_update(self, func):
        """
            @brief  Registers a callback function for notification when the time is updated.

            @param  self    this instance
            @param  func    function call when the time is updated

            @return  a callback ID or cid to allow pulling the callback when no longer necessary.
        """

        cid = self.next_cid
        self.next_cid += 1

        self.callbacks[cid] = func
        return(cid)