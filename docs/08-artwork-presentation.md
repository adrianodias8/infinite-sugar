# Presentation

Each fresh page asks the visitor to name the fly before loading the simulation. About opens
when loading completes; it contains the premise, technical summary, credits and further
reading. Inspect provides neural activity and simulation statistics without editing controls;
its Senses table shows the loop for each sense — the world's level, the switch's level and the
population's rate side by side — so a viewer can see the terrarium reach the neurons.

The status line under the fly is a sentence from the state it is in and what the world offers
("Walking toward the sugar", "Feeding at the sack", "Flying from a threat", "Grooming the
antennae", "Smelling the sugar"), not a state name. Sound is off by default: a button turns on
a wingbeat tone whose loudness follows the stroke and a soft tick as each tripod of feet lands,
both synthesised from the same state that draws the wings and legs. The camera is a spectator:
it stays where the visitor put it, keeps the fly in view by turning, and when the terrarium's
opaque geometry hides the fly it slides along its orbit to the nearest clear angle at a bounded
rate (one raycast every sixth frame; the visitor's own drag still wins).

The hand in the terrarium: a tap on the fly is a touch on that flank; a drag on the beach ball
picks it up. The ball stays a dynamic body while held — each frame its velocity is set to carry
it to where the pointer points on the plane through its centre, capped at 4 cm/s — so it still
collides with the fly and the terrain, and on release it keeps the hand's last speed (up to
8 cm/s), so it can be rolled at the fly, thrown past it or set down. What the fly makes of it
is the world's business: the ball looms by its real approach on the eye it approaches, bumps
as a touch on that flank, and is seen by the eyes. Measured in headless Chromium: a ball
rolled at the fly from 0.6 cm raises the looming level to 0.87 and the escape drive to 173 Hz,
and the fly takes off with the ball still 0.47 cm away (an earlier proxy as wide as the
wingspan kept the ball too far out to alarm it, see [13-world.md](13-world.md)); a slow roll
that reaches the fly reads as a touch of 1.0 on that flank. The ball moves in the fly's own
time, so a throw looks slow when the machine runs below real time. The sugar sack can be
dragged too: it slides along the floor to the pointer and snaps to the nearest spot its
footprint fits that is not under the fly, the floor map is re-stamped, and the odour plume and
the fly's target move with it. The speed button asks for 1×, 2× or 4× simulated time per real second; the
achieved factor is in Inspect.

Sugar starts enabled after resting calibration. The counter accumulates feeding motor spikes
while sugar is enabled; it holds during pause or when sugar is off. The neural map highlights
sampled firing neurons and their outgoing connections. Sensory glow increases only while sugar
is enabled.

The fly's name appears in the page heading, browser title and explanatory text. Exit displays a
confirmation before discarding the simulation. Closing or reloading the tab uses normal browser
behavior, with no unload warning. Neither the name nor the simulation state is persisted.

The interface uses system fonts, native dialogs and responsive layouts. Rendering resolution,
shadows and refresh rates adapt to device performance; see [performance notes](10-mobile-performance.md).
