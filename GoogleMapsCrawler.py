#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# (C) 2018 Felix Feldmann, Conrad Sachweh, Frank Gabel

"""NAME
        GoogleMapsCrawler - <description>

SYNOPSIS
        %(prog)s [--help]

DESCRIPTION
        A script which when given a longitude, latitude and zoom level downloads a
        high resolution google map.

FILES
        none

SEE ALSO
        Adapted from: https://gist.github.com/eskriett/6038468

DIAGNOSTICS
        none

BUGS
        none

AUTHOR
        (Adapted from) Hayden Eskriett [http://eskriett.com]
        Felix Feldmann, felix.feldmann@iwr.uni-heidelberg.de
        Conrad Sachweh, conrad.sachweh@iwr.uni-heidelberg.de
        Frank Gabel,
"""


import urllib
import urllib.request
from PIL import Image
import os
import math

GERMANY = {'north': 53.385433, 'east': 14.394068,
           'west': 6.373989, 'south': 47.439950}



#-------------------------------------------------------------------------------

class GoogleMapDownloader:
    """
        A class which generates high resolution google maps images given
        a longitude, latitude and zoom level
    """

    def __init__(self, lat, lng, zoom=12):
        """
            GoogleMapDownloader Constructor

            Args:
                lat:    The latitude of the location required
                lng:    The longitude of the location required
                zoom:   The zoom level of the location required, ranges from 0 - 23
                        defaults to 12
        """
        self._lat = lat
        self._lng = lng
        self._zoom = zoom

    def getXY(self):
        """
            Generates an X,Y tile coordinate based on the latitude, longitude
            and zoom level

            Returns:    An X,Y tile coordinate
        """

        tile_size = 256

        # Use a left shift to get the power of 2
        # i.e. a zoom level of 2 will have 2^2 = 4 tiles
        numTiles = 1 << self._zoom

        # Find the x_point given the longitude
        point_x = (tile_size/ 2 + self._lng * tile_size / 360.0) * numTiles // tile_size

        # Convert the latitude to radians and take the sine
        sin_y = math.sin(self._lat * (math.pi / 180.0))

        # Calulate the y coorindate
        point_y = ((tile_size / 2) + 0.5 * math.log((1+sin_y)/(1-sin_y)) * -(tile_size / (2 * math.pi))) * numTiles // tile_size

        return int(point_x), int(point_y)

    def generateImage(self, **kwargs):
        """
            Generates an image by stitching a number of google map tiles together.

            Args:
                start_x:        The top-left x-tile coordinate
                start_y:        The top-left y-tile coordinate
                tile_width:     The number of tiles wide the image should be -
                                defaults to 5
                tile_height:    The number of tiles high the image should be -
                                defaults to 5
            Returns:
                A high-resolution Goole Map image.
        """

        start_x = kwargs.get('start_x', None)
        start_y = kwargs.get('start_y', None)
        tile_width = kwargs.get('tile_width', 5)
        tile_height = kwargs.get('tile_height', 5)

        # Check that we have x and y tile coordinates
        if start_x == None or start_y == None :
            start_x, start_y = self.getXY()

        # Determine the size of the image
        width, height = 256 * tile_width, 256 * tile_height

        #Create a new image of the size require
        map_img = Image.new('RGB', (width,height))
        sat_img = Image.new('RGB', (width,height))

        for x in range(0, tile_width):
            for y in range(0, tile_height) :
                if True:
                    if args.label:
                        # Store the image with labels
                        url = 'https://mt0.google.com/vt/lyrs=y&?x=' + str(start_x + x) + '&y=' + str(start_y + y) + '&z=' + str( self._zoom)
                        if args.debug: print(url)
                    else:
                        url = 'https://mt0.google.com/vt/lyrs=s&?x=' + str(start_x + x) + '&y=' + str(start_y + y) + '&z=' + str( self._zoom)
                        if args.debug: print(url)
                    current_tile = str(x)+'-'+str(y)
                    urllib.request.urlretrieve(url, current_tile)

                    im = Image.open(current_tile)
                    sat_img.paste(im, (x*256, y*256))

                    os.remove(current_tile)


                if True:
                    if args.label:
                        url = 'https://mt0.google.com/vt?x='+str(start_x+x)+'&y='+str(start_y+y)+'&z='+str(self._zoom)
                        if args.debug: print(url)
                    else:
                        url = 'https://mt0.google.com/vt?x='+str(start_x+x)+'&y='+str(start_y+y)+'&z='+str(self._zoom) # work needs to be done
                        if args.debug: print(url)

                    current_tile = str(x)+'-'+str(y)
                    urllib.request.urlretrieve(url, current_tile)

                    im = Image.open(current_tile)
                    map_img.paste(im, (x*256, y*256))

                    os.remove(current_tile)

        return map_img, sat_img



#-------------------------------------------------------------------------------


if __name__ == '__main__':
    import argparse
    import sys

    # Command line option parsing example
    parser = argparse.ArgumentParser()
    # Note that -h (--help) is added by default and uses the help strings of
    # the other options. The variables containing the options are automatically
    # created.

    # Boolean option (default is !action)
    parser.add_argument('-q', '--quiet', action='store_true', dest='quiet',
                        help="Don't print status messages to stdout.")
    # Option
    parser.add_argument('-l', dest='label', action='store_true',
                        help='Set to have labels on the images or not.')

    parser.add_argument('-z', dest='zoom_factor', type=int, default=15,
                        help='Zoom level (0-23)')

    parser.add_argument('-d', '--debug', dest='debug', action='store_true',
                        help='Debug Mode.')

    #parser.add_argument('-d', '--debug', dest='debug', action='store_true',
    #                    help='Debug Mode.')

    args = parser.parse_args()

    # -------------------------------------------------------------------------------

    print("Creating Tiles for Germany.")

    # ZOOM STAGES:  13, 15, 19


    # Create a new instance of GoogleMap Downloader
    gmd = GoogleMapDownloader(51.1657, 10.4515, args.zoom_factor)

    print("The tile coorindates are {}".format(gmd.getXY()))

    try:
        # Get the high resolution image
        map, sat = gmd.generateImage()
    except IOError:
        print("Could not generate the image - try adjusting the zoom level and checking your coordinates.")
    else:
        #Save the image to disk
        map.save("map_image_zoom{}.png".format(args.zoom_factor))
        sat.save("sat_image_zoom{}.png".format(args.zoom_factor))
        #print("The map has successfully been created")