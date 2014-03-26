Identify: Mesh Identifier
=========================
Say we have two meshes. One is called Destination (or *D*, for short) and the
other is called Source (or *S*, for short).

We want to figure out the most likely bending, stretching, and rigid moving to
apply to *S* to make it look like *D*. We can imagine that we have *S* lying
around, and while we weren't looking, somebody came and moved it until it
looked like *D*. We have a 3D image of *S* before our coworker messed it up,
and we want to use this to figure out precisely what our coworker did. 

To get some handle on the problem, we'll assume our coworker didn't have time
to take *S* to the moon and back. Instead, we propose, this person probably
did the least amount of effort possible to move *S* to *D*, so we'll try and
recover their motion by finding a least-energy way to move *S* to *D*. This
motion can be expressed as a bijective point mapping that moves each point on
*S* to a corresponding point on *D*. Let's call such a mapping *f*.

Further, let's suppose we're not interested in this entire mapping equally,
but we care about the positions of a few token points more than the rest of
the points on the meshes. In particular, we want to be highly confident that
this mapping gives us a correct correspondence of a subset of points *Q* on
*S* to their partners on *D*, even if that means that the rest of the points
may be less correctly mapped.

Point Set File Format
---------------------
For ease of use, point sets used as input will be given as files with the
following format:
```
<label> <x-coord> <y-coord> <z-coord>
<label> <x-coord> <y-coord> <z-coord>
...
```
`<label>` is a string label with no spaces and no quotation marks, used to
identify this particular point for humans. Output will preserve this labeling.

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
./identify.py [flags] <path to .obj of S model> <path to .obj of D model> <path to point set Q> <path to output point set file>
```
Source and destination files should be in standard `.obj` format. Point set
*Q* should be in the format specified above, and the output file will be in
the output format specified above.

Flags
-----

Flag|Use
----|---
-h:|prints this help message, then exits.
-m:|mesh output mode - instead of a point set file, save the entire cloud under the mapping as a .obj file to &lt;output&#96;file&gt;
-i [s-index] [d-index]:|ensure that the mapping identifies the int of index s-index on S and of d-index on D. By default, the centers of mass of S and D are identified.
-v:|verbose output mode
--convergence=[val]:|run identification until matching confidence exceeds val (default: 0.1)
--algorithm=[val]:|use specified algorithm to perform registration (default: icp).  valid options: icp

Examples
--------
```
./identify.py maya_shorts.obj kinect_shorts.obj Q.txt output.txt
```
