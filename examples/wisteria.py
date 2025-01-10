from typing import List

import taichi as ti
import taichi.math as tm

ti.init(arch=ti.cuda)

import dfractal as df
import dfractal.variations as dfv


def main():
    pipe = FlamePipeline(
        train_n_threads=1000,
        train_n_iters=1000,
        eval_n_threads=100000,
        eval_n_iters=100000,
        train_resolution=(200, 200),
        eval_resolution=(1200, 1200),
    )

    # Set Up
    true_image = df.read_image("./images/Wisteria.jpg", pipe.train_resolution)
    pipe.mse.set_reference_image(true_image)
    pipe.forward()
    initial_image = pipe.train_output_buffer.to_numpy()

    # Visualize
    df.plot_image(true_image, title="true image")
    df.plot_image(initial_image, title="initial image")
    df.show_plots()

    # Train
    window = ti.ui.Window("Live Training Progress", res=pipe.train_resolution)
    canvas = window.get_canvas()
    canvas.set_image(pipe.train_output_buffer)
    window.show()

    # video_manager = ti.tools.VideoManager(
    #     output_dir="./results/wisteria/video",
    #     framerate=24,
    #     automatic_build=False,
    # )

    i = 0
    while window.running:
        i += 1
        pipe.forward()
        canvas.set_image(pipe.train_output_buffer)
        window.show()

        if i % 10 == 0:
            loss = pipe.mse.loss[None]
            print("Train step: {}, loss: {}".format(i, loss))
            # video_manager.write_frame(pipe.train_output_buffer.to_numpy())

        pipe.backward()
        pipe.update(i > 100)

    # Visualize
    final_image = pipe.train_output_buffer.to_numpy()
    df.plot_image(true_image, title="true image")
    df.plot_image(final_image, title="final image")
    df.show_plots()

    # Generate Eval Image
    print("Generating Full Resolution Image")
    pipe.forward_eval()
    eval_image = pipe.eval_output_buffer.to_numpy()
    df.plot_image(true_image, title="true image")
    df.plot_image(eval_image, title="eval image")
    df.show_plots()

    # Save Results
    # video_manager.make_video(mp4=True, gif=True)
    df.save_image("./results/wisteria/result.png", eval_image)


class FlamePipeline(df.Pipeline):
    def __init__(
        self,
        train_n_threads: int,
        train_n_iters: int,
        eval_n_threads: int,
        eval_n_iters: int,
        train_resolution: int,
        eval_resolution: int,
    ):

        self.train_resolution = train_resolution
        self.eval_resolution = eval_resolution

        self.n_samplers = 4

        self.samplers = [
            df.LinearFlameSampler(
                n_generators=24,
                train_n_threads=train_n_threads,
                train_n_iters=train_n_iters,
                eval_n_threads=eval_n_threads,
                eval_n_iters=eval_n_iters,
            )
            for _ in range(self.n_samplers)
        ]

        self.splatters = [
            df.Splatter(
                sampler=self.samplers[i],
                train_resolution=train_resolution,
                eval_resolution=eval_resolution,
            )
            for i in range(self.n_samplers)
        ]
        self.palettes = [
            df.Palette(n_qualities=self.samplers[i].n_generators)
            for i in range(self.n_samplers)
        ]
        self.painters = [
            df.Painter(splatter=self.splatters[i], palette=self.palettes[i])
            for i in range(self.n_samplers)
        ]
        self.compositor = df.Compositor(painters=self.painters)

        self.train_output_buffer = self.compositor.train_output_buffer
        self.eval_output_buffer = self.compositor.eval_output_buffer

        self.mse = df.MeanSquaredError(self.train_output_buffer)

    def update(self, update_sampler: bool):
        for i in range(self.n_samplers):
            self.palettes[i].update(lr=1e3)

            if update_sampler:
                self.splatters[i].update(lr=5e-2)
                self.samplers[i].update(lr=3e-0)

        self.compositor.update(lr=1e-0)


if __name__ == "__main__":
    main()
