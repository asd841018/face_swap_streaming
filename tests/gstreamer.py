import sys
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject

# Initialize GStreamer
Gst.init(None)

def on_pad_added(element, pad, data):
    """
    Callback function called when a new pad is added to the demuxer.
    This dynamically links the demuxer's output to the appropriate decoder/parser.
    """
    print(f"New pad '{pad.get_name()}' added by '{element.get_name()}'")
    
    # Get the pad's capabilities to determine what kind of data it carries
    caps = pad.get_current_caps()
    if caps is None:
        caps = pad.query_caps(None)
    
    structure = caps.get_structure(0)
    name = structure.get_name()
    
    # Link to a parser/decoder if it's a video or audio stream
    if name.startswith("video/"):
        # We assume H264 for common RTMP streams, but you might need to adjust
        sinkpad = data.get_static_pad("sink")
        if sinkpad.is_linked():
            print("Sink pad already linked, skipping.")
        else:
            if pad.link(sinkpad) != Gst.PadLinkReturn.OK:
                print("Could not link video pad")
            else:
                print("Linked video pad successfully")
        sinkpad.unref()
    elif name.startswith("audio/"):
        # Handle audio linking similarly if needed
        pass

def create_pipeline(rtmp_url):
    """
    Creates and runs the GStreamer pipeline for RTMP reception.
    """
    pipeline = Gst.Pipeline.new("rtmp-receiver")

    # Create elements
    source = Gst.ElementFactory.make("rtmpsrc", "source")
    demuxer = Gst.ElementFactory.make("flvdemux", "demuxer")
    h264parse = Gst.ElementFactory.make("h264parse", "h264-parser")
    decoder = Gst.ElementFactory.make("avdec_h264", "h264-decoder")
    converter = Gst.ElementFactory.make("videoconvert", "converter")
    sink = Gst.ElementFactory.make("autovideosink", "sink")

    if not all([source, demuxer, h264parse, decoder, converter, sink, pipeline]):
        print("Not all elements could be created. Check your GStreamer installation.")
        sys.exit(1)

    # Set the RTMP stream location
    source.set_property("location", rtmp_url)

    # Add elements to the pipeline
    pipeline.add(source)
    pipeline.add(demuxer)
    pipeline.add(h264parse)
    pipeline.add(decoder)
    pipeline.add(converter)
    pipeline.add(sink)

    # Link elements (static links)
    if not source.link(demuxer):
        print("ERROR: Could not link source to demuxer")
        sys.exit(1)

    # Link the remaining elements
    if not Gst.Element.link_many(h264parse, decoder, converter, sink):
         print("ERROR: Could not link parser, decoder, converter, and sink")
         sys.exit(1)

    # Connect the pad-added signal to the demuxer
    demuxer.connect("pad-added", on_pad_added, h264parse)

    # Start the pipeline
    pipeline.set_state(Gst.State.PLAYING)
    print(f"Playing from: {rtmp_url}")

    # Listen for messages on the bus
    bus = pipeline.get_bus()
    while True:
        msg = bus.timed_pop_filtered(
            Gst.CLOCK_TIME_NONE,
            Gst.MessageType.ERROR | Gst.MessageType.EOS
        )

        if msg:
            if msg.type == Gst.MessageType.ERROR:
                err, debug_info = msg.parse_error()
                print(f"Error received: {err.message}")
                print(f"Debugging info: {debug_info if debug_info else 'none'}")
                break
            if msg.type == Gst.MessageType.EOS:
                print("End of stream reached")
                break
    
    # Free up resources
    pipeline.set_state(Gst.State.NULL)
    print("Pipeline stopped and resources freed.")

if __name__ == "__main__":
    # Replace with your actual RTMP stream URL
    rtmp_location = "rtmp://localhost:1935/live/myStream" 
    create_pipeline(rtmp_location)