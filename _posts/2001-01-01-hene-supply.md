---
layout: portfolio
title: Universal HeNe power supply
categories: [PCB, Project]
excerpt: This is a universal HeNe power supply with multiple boards designed in KiCad. It allows for high-voltage operation and control of the laser.
is_work: true
---

This is a universal HeNe power supply designed in KiCad. The main goal of this project was to create a high-voltage power supply that can be used for driving HeNe lasers since manufacturer's power supplies are often expensive and hard to repair. This design includes multiple boards to handle different aspects of the power supply, including high-voltage generation, regulation, and control.

## Design

### Controller Board
The controller board is responsible for managing the overall operation of the power supply, including voltage regulation and safety features. It uses an RP2350 microcontroller for control and monitoring, along with various gate drivers, amplifiers and mosfets to handle DC-DC and AC-DC conversion.
<img src="/images/hene-supply/controller-design.png">

### HV Board
The high-voltage board is responsible for generating the high voltage required to drive the HeNe laser. It uses a combination of transformers, rectifiers, and voltage multipliers to achieve the necessary voltage levels.
<img src="/images/hene-supply/hv-design.png">

### User Interface Board
The user interface board provides a way for the user to interact with the power supply, including buttons and a display.
<img src="/images/hene-supply/ui-design.png">

## Manufacturing
The PCBs were manufactured using a professional PCB manufacturing service. The components were soldered using a reflow soldering process to ensure proper connections and reliability also done by the manufacturer.

<img src="/images/hene-supply/man1.png">
<img src="/images/hene-supply/man2.png">
<img src="/images/hene-supply/man3.png">

## Testing
The power supply was tested with a Melles Griot 15mW HeNe laser tube. The results showed that the power supply was able to drive the laser tube successfully.

<img src="/images/hene-supply/test1.png">
<img src="/images/hene-supply/test2.png">

On a side note, the results of this project were presented at the LXVIII Congreso Nacional de Física in Toluca, México. Since this project was born out of a collaborative effort of the Electronics Faculty and Physics Faculty of the Benemérita Universidad Autónoma de Puebla (BUAP), the presentation highlighted the interdisciplinary nature of the work and its potential applications in both educational and research settings.