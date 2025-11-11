---
layout: portfolio
title: Planar Magnetics
categories: [PCB]
excerpt: This is a six layer planar transformer PCB designed in KiCad for use in 1MHz - 5MHz frequencies.
is_work: true
---

This is a six layer 2:1 planar transformer PCB designed in KiCad for use in 1MHz - 5MHz frequencies. The main goal of this project was to create a small transformer that can be used in a variety of applications, including power supplies and RF circuits. The design was driven by the need for a compact and efficient transformer that could operate at high frequencies without excessive losses. Also I had a coupon with the manufacturer that allowed me to create a six layer PCB at a very low cost, so I decided to take advantage of that to create this planar transformer with some "unordinary" design choices like 2oz copper on all layers.

## Design

The PCB was designed using KiCad with the help of this website: https://webench.ti.com/wb5/LDC/#/spirals in order to calculate the spirals.
<img src="/images/planar-magnetics/design.png">

## Manufacturing
The PCB was manufactured using a professional PCB manufacturing service. six layer PCB with 2oz copper on all layers.

<img src="/images/planar-magnetics/preview.png">

## Testing

A 1.5MHz sine wave is applied to the primary side of the transformer and the output is measured on the secondary side. The results show that the transformer is able to step up the voltage as expected, with a measured output voltage of approximately 3.8 times the input.

<img src="/images/planar-magnetics/test1.png">
<img src="/images/planar-magnetics/test2.png">
<img src="/images/planar-magnetics/test3.png">