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

Sugar starts enabled after resting calibration. The counter accumulates feeding motor spikes
while sugar is enabled; it holds during pause or when sugar is off. The neural map highlights
sampled firing neurons and their outgoing connections. Sensory glow increases only while sugar
is enabled.

The fly's name appears in the page heading, browser title and explanatory text. Exit displays a
confirmation before discarding the simulation. Closing or reloading the tab uses normal browser
behavior, with no unload warning. Neither the name nor the simulation state is persisted.

The interface uses system fonts, native dialogs and responsive layouts. Rendering resolution,
shadows and refresh rates adapt to device performance; see [performance notes](10-mobile-performance.md).
