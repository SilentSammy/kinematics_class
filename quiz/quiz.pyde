from text_helpers import Text, queue_text, draw_queued_text
import random

def setup():
    fullScreen(P3D)
    lights()

def draw():
    pushMatrix()
    setupScene()
    draw3DStuff()
    popMatrix()
    
    # Draw all queued text in screen space
    slide_counter = "Slide {}/{}".format(current_slide + 1, len(slides))
    Text(slide_counter, 24, 0.01, 0.94, col=(128, 128, 128)).draw()
    Text("Use arrow keys to navigate slides", 20, 0.01, 0.97, col=(128, 128, 128)).draw()
    draw_queued_text()

def draw3DStuff():
    # Call the current slide function
    slides[current_slide]()

# SLIDES
def sectionTitle(title, subtitle=""):
    queue_text(Text(title, 48, 0.5, 0.45, col=(0, 0, 0), h_align=CENTER, v_align=CENTER))
    if subtitle:
        queue_text(Text(subtitle, 32, 0.5, 0.55, col=(100, 100, 100), h_align=CENTER, v_align=CENTER))

def missingAxisQuestions():
    def question(q_idx, rotX, rotY, missing_idx):
        queue_text(Text("{}. Where will the missing axis point?".format(q_idx), 32, 0.05, 0.05, col=(0, 0, 0)))
        # Rotate view to throw off the student
        rotateX(rotX)
        rotateY(rotY)
        # Reference frame - hide the specified axis
        drawLines([axes[i] for i in range(3) if i != missing_idx])

    def answer(q_idx, rotX, rotY):
        queue_text(Text("{}. Answer".format(q_idx), 32, 0.05, 0.05))
        # Same rotation as question
        rotateX(rotX)
        rotateY(rotY)
        drawAxes()

    # Generate random questions
    axis_pool = [0, 1, 2]
    random.shuffle(axis_pool)
    
    questions = []
    
    # First 2 questions: random missing axis, no rotation
    for i in range(2):
        missing_idx = axis_pool[i]
        questions.append((0, 0, missing_idx))
    
    # Next 3 questions: completely random
    random.shuffle(axis_pool)
    for i in range(3):
        rotX = random.uniform(-PI/2, PI/2)
        rotY = random.uniform(-PI/2, PI/2)
        missing_idx = axis_pool[i % 3]  # Cycle through pool if needed
        questions.append((rotX, rotY, missing_idx))

    # Generate 6 functions: 3 missing axis questions and 3 answers
    slide_funcs = []
    slide_funcs.append(lambda: sectionTitle("Missing Axis Questions", "Identify the missing coordinate axis"))
    
    for i, (rotX, rotY, missing_idx) in enumerate(questions):
        q_idx = i + 1
        # Use default arguments to capture loop variables (avoid late binding)
        slide_funcs.append(lambda rx=rotX, ry=rotY, mi=missing_idx, qi=q_idx: question(qi, rx, ry, mi))
        slide_funcs.append(lambda rx=rotX, ry=rotY, qi=q_idx: answer(qi, rx, ry))
    
    return slide_funcs

def translationQuestions():
    def question(q_idx, x, y, z, drawAx=True, objectPath=None):
        queue_text(Text("{}. How will the object move if translated by ({}, {}, {})?".format(q_idx, x, y, z), 32, 0.05, 0.05, col=(0, 0, 0)))
        if drawAx:
            drawAxes()
        if objectPath:
            drawObject(objectPath)
        else:
            drawBox()

    def answer(q_idx, x, y, z, objectPath=None):
        queue_text(Text("{}. Translating to ({}, {}, {}) looks like this".format(q_idx, x, y, z), 32, 0.05, 0.05))
        # Draw ghost object/box at starting position
        drawAxes()
        if objectPath:
            drawObject(objectPath, opacity=80)
        else:
            drawBox(opacity=80)
        
        # Draw animated object/box
        animateTranslate(x, y, z, 4.0, 3.0)
        if objectPath:
            drawObject(objectPath)
        else:
            drawBox()
        drawAxes()

    questions = [ # x, y, z translations (optionally: drawAx, objectPath)
        # First 2: single axis, no object
        (10, 0, 0),
        (0, -10, 0),
        # Next 2: introduce object
        (0, 0, 10, True, "model.obj"),
        (-10, 0, 0, True, "model.obj"),
        # Next 2: hide axes
        (0, 10, 0, False, "model.obj"),
        (0, 0, -10, False, "model.obj"),
        # Next 2: show axes again, but now multi-axis
        (10, 10, 0, True, "model.obj"),
        (-10, 0, 10, True, "model.obj"),
        # Next 2: hide axes, multi-axis
        (10, -10, 10, False, "model.obj"),
        (-10, 10, -10, False, "model.obj"),
    ]

    slides = []
    slides.append(lambda: sectionTitle("Translations", "Identify where the object will move"))
    for i, q in enumerate(questions):
        q_idx = i + 1
        # Unpack tuple with optional parameters
        x, y, z = q[0], q[1], q[2]
        drawAx = q[3] if len(q) > 3 else True
        objectPath = q[4] if len(q) > 4 else None
        
        slides.append(lambda xi=x, yi=y, zi=z, da=drawAx, op=objectPath, qi=q_idx: question(qi, xi, yi, zi, da, op))
        slides.append(lambda xi=x, yi=y, zi=z, op=objectPath, qi=q_idx: answer(qi, xi, yi, zi, op))
    
    return slides

def rotationQuestions():
    def question(q_idx, axis, angle_deg, drawAx=True, objectPath=None):
        axis_names = ['X', 'Y', 'Z']
        queue_text(Text("{}. How will the object move if rotated {} deg about the {}-axis?".format(q_idx, int(angle_deg), axis_names[axis]), 32, 0.05, 0.05, col=(0, 0, 0)))
        if drawAx:
            drawAxesHighlight(axis)
        if objectPath:
            drawObject(objectPath)
        else:
            drawBox()

    def answer(q_idx, axis, angle_deg, objectPath=None):
        axis_names = ['X', 'Y', 'Z']
        queue_text(Text("{}. Rotating {} deg about the {}-axis looks like this".format(q_idx, int(angle_deg), axis_names[axis]), 32, 0.05, 0.05))
        # Draw ghost object/box at starting position
        drawAxesHighlight(axis)
        if objectPath:
            drawObject(objectPath, opacity=80)
        else:
            drawBox(opacity=80)
        
        # Draw animated object/box
        animateRotate(axis, radians(angle_deg), 4.0, 3.0)
        if objectPath:
            drawObject(objectPath)
        else:
            drawBox()
        drawAxesHighlight(axis)

    questions = [ # axis (0=X, 1=Y, 2=Z), angle in degrees (optionally: drawAx, objectPath)
        # First 2: single axis, small angles, no object
        (0, 90),
        (1, -90),
        # Next 2: introduce object
        (2, 90, True, "model.obj"),
        (0, -90, True, "model.obj"),
        # Next 2: hide axes
        (1, 180, False, "model.obj"),
        (2, -180, False, "model.obj"),
        # Next 2: show axes again, larger angles
        (0, 180, True, "model.obj"),
        (1, 270, True, "model.obj"),
        # Next 2: hide axes, challenging angles
        (2, -270, False, "model.obj"),
        (0, 45, False, "model.obj"),
    ]

    slides = []
    slides.append(lambda: sectionTitle("Rotations", "Identify how the object will rotate"))
    for i, q in enumerate(questions):
        q_idx = i + 1
        # Unpack tuple with optional parameters
        axis = q[0]
        angle_deg = q[1]
        drawAx = q[2] if len(q) > 2 else True
        objectPath = q[3] if len(q) > 3 else None
        
        slides.append(lambda a=axis, ang=angle_deg, da=drawAx, op=objectPath, qi=q_idx: question(qi, a, ang, da, op))
        slides.append(lambda a=axis, ang=angle_deg, op=objectPath, qi=q_idx: answer(qi, a, ang, op))
    
    return slides


# SLIDE STUFF
slides = [
    # missingAxisQuestions(),
    # translationQuestions(),
    rotationQuestions(),
]
slides = [item for sublist in slides for item in sublist]  # Flatten list of lists
current_slide = 0

# DRAWING HELPERS
def animate(duration, offset=0):
    """Return normalized time (0.0 to 1.0) over duration seconds, with optional offset."""
    time = millis() / 1000.0
    adjusted_time = (time - offset) % duration
    return adjusted_time / duration

def animateSmooth(duration, offset=0):
    """Return smoothed normalized time (0.0 to 1.0) with cosine easing."""
    t = animate(duration, offset)
    return (1 - cos(t * PI)) / 2.0

def animateHold(cycle_duration, anim_duration, offset=0):
    """Animate from 0.0 to 1.0 over anim_duration, then hold at 1.0 for rest of cycle."""
    return min(animate(cycle_duration, offset) * (cycle_duration / anim_duration), 1.0)

def animateHoldSmooth(cycle_duration, anim_duration, offset=0):
    """Smoothly animate from 0.0 to 1.0 over anim_duration with easing, then hold at 1.0."""
    time = millis() / 1000.0
    t = ((time - offset) % cycle_duration)
    if t >= anim_duration:
        return 1.0
    normalized = t / anim_duration
    return (1 - cos(normalized * PI)) / 2.0

def animateTranslate(x, y, z, cycle_duration, anim_duration, offset=0):
    """Translate from origin to (x, y, z) over anim_duration, then hold."""
    factor = animateHoldSmooth(cycle_duration, anim_duration, offset)
    translate(x * factor, y * factor, z * factor)

def animateRotate(axis, angle, cycle_duration, anim_duration, offset=0):
    """Rotate from 0 to angle about specified axis over anim_duration, then hold."""
    factor = animateHoldSmooth(cycle_duration, anim_duration, offset)
    if axis == 0:  # X-axis
        rotateX(angle * factor)
    elif axis == 1:  # Y-axis
        rotateY(angle * factor)
    elif axis == 2:  # Z-axis
        rotateZ(angle * factor)

def drawLines(lines):
    for weight, col, line_coords in lines:
        strokeWeight(weight)
        stroke(*col)
        line(*line_coords)

def drawAxes():
    drawLines(axes)

def drawAxesHighlight(highlight_axis=None):
    """Draw axes with optional highlighting of one axis."""
    for i, (weight, col, line_coords) in enumerate(axes):
        if highlight_axis is not None and i == highlight_axis:
            # Highlighted axis: thicker and brighter
            strokeWeight(0.3)
            stroke(col[0], col[1], col[2])
        else:
            # Normal axis
            strokeWeight(weight)
            stroke(*col)
        line(*line_coords)

def drawBox(opacity=255):
    strokeWeight(0.1)
    stroke(0)
    fill(200, 100, 100, opacity)
    box(4, 4, 4)

# Lazy-load and draw a 3D object from a file path
objects = {}  # Cache for loaded objects
def drawObject(filepath, opacity=255, scaleSize=1.0, rotX=0, rotY=0, rotZ=PI):
    if filepath not in objects:
        # Load the object only if it hasn't been loaded yet
        objects[filepath] = loadShape(filepath)
    
    # Draw the object with the specified opacity, scale, and rotation
    obj = objects[filepath]
    pushMatrix()
    scale(scaleSize)
    rotateX(rotX)
    rotateY(rotY)
    rotateZ(rotZ)

    shape(obj)

    popMatrix()

axes = [
    (0.1, (255, 0, 0), (0, 0, 0, 10, 0, 0)),  # X-axis: red, goes to (10,0,0)
    (0.1, (0, 255, 0), (0, 0, 0, 0, 10, 0)),  # Y-axis: green, goes to (0,10,0)
    (0.1, (0, 0, 255), (0, 0, 0, 0, 0, 10)),  # Z-axis: blue, goes to (0,0,10)
]

# SCENE STUFF
angleX = -0.2
angleY = 0.2
zoom = 20

def setupScene():
    background(255)
    translate(width/2, height/2, 0)
    scale(zoom)
    
    rotateX(angleX)
    rotateY(angleY)
    rotateX(PI/2)
    rotateZ(PI/2)

# EVENTS
def mouseDragged():
    global angleX, angleY
    angleY += (mouseX - pmouseX) * 0.01
    angleX -= (mouseY - pmouseY) * 0.01

def mouseWheel(event):
    global zoom
    zoom -= event.getCount() * 2  # Additive for smoother trackpad
    zoom = max(1, min(100, zoom))

def keyPressed():
    global current_slide
    if key == CODED:
        if keyCode == RIGHT or keyCode == DOWN:
            current_slide = min(current_slide + 1, len(slides) - 1)
        elif keyCode == LEFT or keyCode == UP:
            current_slide = max(current_slide - 1, 0)
