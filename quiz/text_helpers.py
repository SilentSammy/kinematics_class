# TEXT HELPERS
text_queue = []

class Text:
    def __init__(self, msg, font_size, x_frac, y_frac, col=(0,0,0), h_align=LEFT, v_align=TOP):
        self.msg = msg
        self.font_size = font_size
        self.x_frac = x_frac
        self.y_frac = y_frac
        self.col = col
        self.h_align = h_align
        self.v_align = v_align
    
    def draw(self):
        textSize(self.font_size)
        textAlign(self.h_align, self.v_align)
        
        # Calculate actual pixel positions from fractions
        x = width * self.x_frac
        y = height * self.y_frac
        
        # Disable depth testing to ensure the text appears over the 3D graphics
        hint(DISABLE_DEPTH_TEST)
        fill(*self.col)
        text(self.msg, x, y)
        hint(ENABLE_DEPTH_TEST)

def queue_text(text_obj):
    global text_queue
    text_queue.append(text_obj)

def draw_queued_text():
    global text_queue
    for txt in text_queue:
        txt.draw()
    text_queue = []
