---
layout: portfolio
title: Differential Probe PCB
categories: [PCB]
excerpt: This is a differential probe PCB designed in KiCad. It allows for measurements of high voltage signals.
is_work: true
---

This is a differential probe PCB designed in KiCad. The main goal of this project was to create a high-precision differential probe that can be used for measuring high voltage signals safely by having different ground references. Also in my power electronics class there werent enough differential probes available for students to use, so I decided to design my own making this project born out of necessity.
## Design

The PCB was designed using KiCad using the INA-128 instrumentation amplifier as the main component for signal conditioning.
<img src="/images/differential-probe/design.png">

## Manufacturing
The PCB was manufactured using a professional PCB manufacturing service. The components were soldered using a reflow soldering process to ensure proper connections and reliability also done by the manufacturer since some of the components were somewhat exotic and was cheaper than having to order them separately.

<img src="/images/differential-probe/man1.png">

<img src="/images/differential-probe/preview.png">

## Testing
The differential probe was tested with a classic SRC light bulb brightness control circuit. The results showed that the probe was able to accurately measure the differential voltage across the load, allowing for safe and precise measurements of high voltage signals.

<img src="/images/differential-probe/test1.png">
<img src="/images/differential-probe/test2.png">