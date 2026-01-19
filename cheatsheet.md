# Basic shapes
- `box(size)` or `box(width, height, depth)` - draws a 3D box
- `sphere(radius)` - draws a 3D sphere
- `line(x1, y1, z1, x2, y2, z2)` - draws a line between two points

# Formatting
- `fill(r, g, b)` or `fill(r, g, b, alpha)` - sets fill color for shapes
- `stroke(r, g, b)` - sets line/edge color
- `strokeWeight(thickness)` - sets line thickness

# Transforms
- `translate(x, y, z)` - moves the coordinate system
- `rotateX(angle)`, `rotateY(angle)`, `rotateZ(angle)` - rotates around axis (angle in radians)
- `pushMatrix()` - saves current transformation state
- `popMatrix()` - restores to last saved transformation state
