# Thermal Conduction Demo

This demo solves a camera-driven thermal conduction problem on a detailed Mars
rover surface. The rover mesh is complex and imperfect, making it difficult to
volume mesh without substantial geometry repair; WoSX instead works directly
with the surface geometry and geometric queries.

For a given camera view, rays are traced through the image and visible hits on
the rover surface become the sample points. The solution is written back into
image space, producing a thermal rendering from the camera.

<div align="center">

<table>
  <tr>
    <th>Camera views</th>
    <td align="center"><img src="data/rover_view_0.png" alt="Rover camera view 0" height="180"></td>
    <td align="center"><img src="data/rover_view_3.png" alt="Rover camera view 3" height="180"></td>
    <td align="center"><img src="data/rover_view_4.png" alt="Rover camera view 4" height="180"></td>
  </tr>
  <tr>
    <th>Thermal renderings</th>
    <td align="center"><img src="solutions/solution_0_color.png" alt="Thermal rendering for view 0" height="180"></td>
    <td align="center"><img src="solutions/solution_3_color.png" alt="Thermal rendering for view 3" height="180"></td>
    <td align="center"><img src="solutions/solution_4_color.png" alt="Thermal rendering for view 4" height="180"></td>
  </tr>
</table>

</div>

## Technical Details

The demo solves a Laplace equation for a scalar steady-state temperature field,
with no volume heat source. The rover surface uses spatially varying Robin
boundary conditions. Both the Robin value and Robin coefficient are loaded from
grayscale texture maps, queried through the mesh UV parameterization:
`data/robin_value.png` and `data/robin_coefficient.png`.

The mesh is registered as a reflecting Robin boundary, while a surrounding box
provides the auxiliary absorbing boundary used by the walk on stars solver. The
camera file stores the Polyscope view matrix and field of view. The app
generates one ray per image pixel, intersects those rays with the rover, keeps
only visible surface hits, and estimates the solution at those boundary sample
points. Pixels without a visible hit remain background.

The main output is:

- `Solution`: a grayscale image and optional colormapped thermal rendering in
  camera space.

Relevant settings live in `config.json`: `problem.geometry`,
`problem.robinValueTexture`, `problem.robinCoefficientTexture`,
`problem.cameraView`, `problem.cameraWidth`, `problem.cameraHeight`, walk
settings under `solver`, and image/colormap settings under `output`.

**NOTE**: Computing temperatures for a given view is computationally expensive
compared to the lightweight setup visualization. For context, rendering views 0–2
takes approximately 2–3 minutes per view on a Blackwell RTX PRO 6000-class GPU
using default settings; other views will require additional time. This demo
currently struggles with extremely long walk lengths, which remains an open
research challenge.

## Demo Modes

With `output.visualizeSetup` set to `true`, the demo opens a Polyscope viewer
for inspecting the rover mesh, UVs, Robin textures, loaded camera view, and
view-dependent sample points. The GUI also supports generating sample points
from the current view and writing a camera view to file. With `visualizeSetup`
set to `false`, the demo runs the GPU solver and writes the image-space solution
to `output.solutionFile`, plus a colormapped image when `saveColormapped` is
enabled.

## Running the C++ Demo

First, unzip `data/rover.zip`. Then run the executable from the build directory
as follows:

```bash
cd build
./demo_apps/thermal_conduction ../demo_apps/thermal_conduction/config.json
```

When `visualizeSetup` is `false`, images are written relative to
`demo_apps/thermal_conduction/`, for example `solutions/solution_0.png` and
`solutions/solution_0_color.png`.

## Running the Python Demo

Run the Python app from the repository root:

```bash
python demo_apps/thermal_conduction/app.py --config=demo_apps/thermal_conduction/config.json
```

The Python version mirrors the C++ demo structure and uses the same
`config.json` file.
