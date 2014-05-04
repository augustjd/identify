Identify: Mesh Identifier
========================= 
Say we have two meshes. One is called Destination (or _D_, for short) and the other is called Source (or _S_, for short).

We want to figure out the most likely bending, stretching, and rigid moving to
apply to _S_ to make it look like _D_. We can imagine that we have _S_ lying
around, and while we weren't looking, somebody came and moved it until it
looked like _D_. We have a 3D image of _S_ before our coworker messed it up,
and we want to use this to figure out precisely what our coworker did. 

To get some handle on the problem, we'll assume our coworker didn't have time
to take _S_ to the moon and back. Instead, we propose, this person probably
did the least amount of effort possible to move _S_ to _D_, so we'll try and
recover their motion by finding a least-energy way to move _S_ to _D_. This
motion can be expressed as a bijective point mapping that moves each point on
_S_ to a corresponding point on _D_. Let's call such a mapping _f_.

Further, let's suppose we're not interested in this entire mapping equally,
but we care about the positions of a few token points more than the rest of
the points on the meshes. In particular, we want to be highly confident that
this mapping gives us a correct correspondence of a subset of points _Q_ on
_S_ to their partners on _D_, even if that means that the rest of the points
may be less correctly mapped.

Point Set File Format
---------------------
For ease of use, point sets used as input will be given as files with the
following format:
```
<label> <x-coord> <y-coord> <z-coord>
<label> <x-coord> <y-coord> <z-coord>
grasp-points <source-x-coord> <source-y-coord> <source-z-coord> <dest-x-coord> <dest-y-coord> <dest-z-coord>
<label> <x-coord> <y-coord> <z-coord>
<label> <index>
<label> <x-coord> <y-coord> <z-coord>
...
```
`<label>` is a string label with no spaces and no quotation marks, used to
identify this particular point for humans. Output will preserve this labeling.

_NOTE_: `<label>` cannot be `grasp-points`, since that name is reserved for specifying the identity points on the two meshes, as shown below:

`grasp-points` is an optional line used to specify a point in source
coordinates and a point in destination coordinates that will be guaranteed to
be identified by the transformation.

Each point's coordinates can be specified in one of two ways:

`<index>` is the index of the vertex on the source mesh, as an integer

`<_-coord>` is a floating-point number, ideally in scientific format with 6
decimal points of accuracy, but anything that can be parsed by the default
Python float parser is fine.

Newlines should be `'\n'`, and only one point may be entered per line. If
there are any problems parsing the input file, execution will terminate and
the user will be informed of errors in the file.

Output Point Set File Format
----------------------------
After identification, the program will output a point set file with a similar
format.
```
<label> <x-coord> <y-coord> <z-coord> <predicted-x-coord> <predicted-y-coord> <predicted-z-coord> <confidence>
<label> <x-coord> <y-coord> <z-coord> <predicted-x-coord> <predicted-y-coord> <predicted-z-coord> <confidence>
...
```
`<predicted-_-coord>` is a fixed point number with 8 decimal places
corresponding to the predicted position on *D* of this point on *S*.

`<confidence>` is a fixed point number with 8 decimal places between 0.0 and
1.0, given in decimal notation, which corresponds to the accuracy with which
this point is predicted to have been mapped. A score of 0.0 is bad, meaning
that the system has low confidence in this mapping, while a score of 1.0 is
good, meaning that the system believes the mapping is almost exact.

Our algorithm guarantees that the predicted coordinates will correspond to
actual points on *D*, even if the correspondence is incorrect.

Usage
-----
```
./identify.py [flags] <source_file> <destination_file> <point_set_file> <output_file>
```
Source and destination files should be in standard `.obj` format. Point set
*Q* should be in the format specified above, and the output file will be in
the output format specified above.

If you don't need a point set matching, but just want a mesh output, invoke as:
```
./identify.py -no-point-file -m [mesh_output_file] [flags] <source_file> <destination_file>
```

Flags
-----

Flag|Use
----|---
`-h`|prints this help message, then exits.
`-m [mesh_output_file]`|mesh output mode - save the entire cloud under the mapping as a .obj file to `[mesh_output_file]`
`-v`|verbose output mode
`-no-point-file`|Suppress output
`-i [s-index] [d-index]`|ensure that the mapping identifies the point of index s-index on the source mesh and d-index on the destination mesh. By default, centers of mass will be identified.
`--algorithm=[val]`|use specified algorithm to perform registration (default: icp). valid options: icp, radial, curvatureicp, curvature, simple, energy, li
