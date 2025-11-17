---
layout: portfolio
title: LR1121 Transceiver PCB
categories: [PCB]
excerpt: This is an LR1121 transceiver PCB designed in KiCad for long-range low-power communication applications with an integrated antenna.
is_work: true
---

This is an LR1121 transceiver PCB designed in KiCad for long-range low-power communication applications with an integrated antenna. The main goal of this project was to create a compact and efficient transceiver module that can be easily integrated into various IoT devices. Initially this design was meant to use the LR2021 chip but Semtech did not release the chip on time for the project deadline so I had to switch to the LR1121 which was available at the time (and cheaper since there were premade modules out there). Because of that the board design was kept in 6 layers but also because of the antenna geometry requirements of a flat surface.
## Design

The PCB was designed using KiCad using the LR1121 transceiver chip as the main component for long-range communication and the AD8132 differential amplifier for signal conditioning.
<img src="/images/lr1121-transceiver/design.png">

## Antenna
The PCB features an integrated inverted-F antenna designed to operate in the 2400 MHz frequency band. The antenna was simulated and optimized using the electromagnetic solver software Sonnet to ensure optimal performance and efficiency.
<img src="/images/lr1121-transceiver/sim.png">


## Manufacturing
The PCB was manufactured using a professional PCB manufacturing service. The components were soldered using a reflow soldering process to ensure proper connections and reliability also done by the manufacturer since some of the components were pretty much exotic and was a lot cheaper than having to order them separately.

<img src="/images/lr1121-transceiver/man1.png">

<img src="/images/lr1121-transceiver/preview.png">

## Testing
The PCB antenna was tested using a network analyzer to measure its performance and ensure it met the design specifications. The transceiver functionality was verified by establishing communication with another LR1121 module and measuring the signal strength and data transfer rates.

 <video width="100%" height="100%" autoplay loop muted>
  <source src="/images/esp32-diy-relay/test.mp4" type="video/mp4">
Your browser does not support the video tag.
</video> 

