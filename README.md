# HoopTrainer 🏀 — Basketball Shot Tracker

> A single-camera computer vision app that tracks basketball makes and misses in real time — from your phone, in the browser, no extra hardware.


---
<img width="800" height="450" alt="demo_gif" src="https://github.com/user-attachments/assets/49e86179-946d-45ac-8b1e-d1bcdee28b4b" />


## How to run ❗

[hooptrainerv2.netlify.app](https://hooptrainerv2.netlify.app)



Visit this link on your phone, click the "live" icon near the top of the screen and follow instructions from there for camera positioning and setup.

<img width="511" height="762" alt="Screenshot 2026-07-07 at 7 56 14 PM" src="https://github.com/user-attachments/assets/5b16f5c3-e077-4b02-a0a2-877274af796b" />

---

## What it does 🎥 

Simply prop your phone on a tripod, place it on the basketball court, and start letting it fly! This web application will track your makes and misses in real time, utilizing simple computer vision techniques to track the ball and its trajectory.

---

## How it works 💭

The system has two independent components: an **eye** (ball detection) and a **brain** (decides make or miss).

**Ball Detection —** 👀 

Ball detection is split into two main subcomponents. I employed a "blob detector" system that answers the following central question: what's moving, and what's orange? We separately answer these two questions.

For motion detection, we crop a small rectangular region around the basket of interest and each frame we create an array of grayscale values for the pixels inside that region. If the grayscale value of a pixel differs by more than 25 to the benchmark (idle state, updating dynamically over time to adjust for lighting changes), we conclude that there is motion in that pixel. This alone is not able to identify the ball though; other things like the net swaying or lighting changes may cause differences in grayscale values.

For color detection, we first convert the pixels in the area of interest into HSV values instead of the traditional RGB values. We then identify the red/orange ish pixels within the desired rectangular region. However, this alone is also not sufficient to detect the ball since other objects of similar color like the rim or background colors could be in the frame.

We then infer that the overlap between these two arrays (flagged in both the motion and color arrays) to be the basketball. After a bit of cleaning with OpenCV techniques, we identify the best "blobs" as candidates for the center of the ball. Neither indicator alone can reliably predict the location of the ball, but their overlap almost always does.

Occasionally, there might be more than one surviving candidate. This can be caused by the backboard reflecting the ball, skin tones, or maroon colored backgrounds somehow interfering. To resolve ambiguity we keep a running record of the ball's last known position and approximate velocity. We can thus use linear extrapolation to approximate the current position of the ball from its last known position, and we pick the blob that is closest to this approximation and reject candidates that would require the blob to teleport across long distances. If there are no surviving candidates, then we do not report a detection for the current frame, as a missed frame is recoverable (maybe next frame is clearer and the ball gets detected) but a false verdict is not.


<img width="800" height="450" alt="visualization_demo" src="https://github.com/user-attachments/assets/318451e7-28d2-4bb4-a118-22c12580bdd1" />


**Make/miss logic —** 🧠

To determine whether or not a shot is successful or not, we utilize a simple two state system. From the specific camera angle we use (camera placed at the top of the key, facing directly towards the basket) this becomes a 2D coordinate problem. We first map out the relevant parts of the rim with a rectangular box (done by the user when calibrating the setup to ensure maximum accuracy)

<img width="580" height="531" alt="Screenshot 2026-07-07 at 7 22 17 PM" src="https://github.com/user-attachments/assets/b0389c61-48e9-4f2a-bca4-ef18c68d2b93" />


The program starts in an "idle" waiting state, watching for the ball to rise above (or close to) the rim. When it does, the program switches to an "airborne" state and it then waits for the ball to reappear below the net. If the ball comes down inside the rim's horizontal range, the shot is registered as a make. 

<img width="800" height="450" alt="make_demo" src="https://github.com/user-attachments/assets/0d30d63e-509f-4d9c-b27f-2a3a9b80bc2c" />


If it comes down outside the horizontal region, it is deemed a miss. 


<img width="800" height="450" alt="miss_demo" src="https://github.com/user-attachments/assets/23293311-0ef0-46a4-83a0-27253fd4a5c5" />


Two extra checks are used to help refine this algorithm. The first is a net-contact signal that rescues the make when the shot rattles inside and is kicked out sideways. In the shot below, the shot spends significant time in contact with the net (within rectanglular box) and has a downward trajectory, so even if the shot falls slightly outside the horizontal range we count it as a make. Shots that land outside without spending much time in the box/don't have a clear downward trajectory are not saved by this check. (the UI looks slightly different as this is in the original Python backtester and not the HTML port although they are designed to be logically equivalent).

<img width="800" height="450" alt="swivel_demo" src="https://github.com/user-attachments/assets/3ecf8a77-8c0a-4b8a-b6f8-c984a95c3ce1" />


The second is a ball-size (depth) check to catch short misses that bounce back towards the camera. If the ball appears too large, even if it shows up in the horizontal band of the net, it will be revoked to a "miss"


---

## Design decisions 🛠️


**Blob detector over YOLO.** 📐

I started with a YOLO based approach for the 'eye' (ball detection) using a COCO pretrained model. The results weren't great when I backtested against my own shootaround clips. Specifically, the COCO model loses significant accuracy in its tracking when the ball gets close to the rim, often losing track of the ball completely. COCO only has a generic "sports ball" category and does not filter specifically to basketball so it may not be specialized for a basketball at that distance and lighting. Moreover, YOLO's neural net inference takes an approach which is inconsistent across hardware; the same backtest on the same video clip on my local hardware and different sandboxed environments returned different results, specifically for the edge cases including near misses and shots that swivel in the net. In addition, a neural network based YOLO based approach generally has higher processing time than a deterministic alternative which could slow down processing speeds and thus reduce the feasibility of porting it into a HTML based web app running in real time.

The blob detector approach utilizes a rule-based deterministic approach to track the ball that is consistent across hardware. This ensures that any debugging or tuning of the logic would have consistent results on the same video clip no matter the device is it played on. Pixel colors and grayscale values will be the same no matter the device you run the program on. Given more time and resources I will try refining a YOLO based approach starting with just running successful backtests on a computer but I imagine it would be harder to port to a real time web based app.

**Camera Angle.** 📷

I decided to go with a "top of the key" viewpoint for the tripod. This gives the clearest signal and an easy 2D mapping for shots that go in, as the hoop is dead straight on. Shots fly along the camera axis so (most) misses read cleanly left or right. 

**Single camera, phone-first.** 📱

With this project I wanted to emphasize portability and ease of use. Many of the issues I faced (such as the depth issues or front rim misses) could be solved with a much higher elevation point (giving a bird's eye view of each shot) or a second camera angle on the side (with a shot only being registered as a make if both cameras agree on the verdict) but this contradicts my original intent of having an easy setup. The goal was to minimize the equipment required to just 1 phone and tripod.

---

## Known limitations ⚠️

- Shots that are short or miss off the front rim sometimes are falsely registered as a make. This is because the area directly in front of the rim and the interior of the rim share the same area from a 2D top of the key view. I've tried mitigating this with a depth check, ie. that if the ball is detected in that area but is larger than expected, a make will be downgraded to a miss, but this is still a key limitation as a result of this camera angle
- 3 pointers can have inconsistent trajectory tracking due to (plausibly, not entirely sure yet) the higher velocity of the ball
- Missed shots that do not reach a sufficient height threshold are occasionally not detected as shot attempts (ie. the "idle -> airborne" transition is not made). I am currently trying to fix this issue by finding the ideal height threshold for what counts as a "shot"
- Airballs that graze the net without touching the rim are currently falsely flagged as makes. 


## Tech ⚙️

Python + OpenCV for native computer backtesting, with an equivalent HTML implementation for the web based port. The intended effect is to simply run the HTML web app to test the program but you can also manually upload footage and backtest with the Python programs.

---

## Notes 📝

This project was inspired by the Homecourt App and also the release of Claude Fable 5. Built with AI assistance (Claude Fable 5) for implementation, with the architecture, design, detector choice, and validation driven by me.

Btw go Sixers (Maxey the GOAT)
