import taichi as ti
import taichi.math as tm

ti.init(arch=ti.cuda, random_seed=42)

import dfractal as df


def main():
    pipe = FlamePipeline(
        train_n_threads=1000,
        train_n_iters=1000,
        eval_n_threads=100000,
        eval_n_iters=100000,
        train_resolution=(100, 100),
        eval_resolution=(600, 600),
    )

    # Set Up
    true_image = df.read_image("./images/circles3.png", pipe.train_resolution)
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

    i = 0
    while window.running:
        i += 1
        pipe.forward()
        canvas.set_image(pipe.train_output_buffer)
        window.show()

        if i % 10 == 0:
            loss = pipe.mse.loss[None]
            print("Train step: {}, loss: {}".format(i, loss))

        pipe.backward()
        pipe.update()

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

    # df.save_image("./results/linear_flame/result.png", eval_image)


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

        self.samplers = [
            df.LinearFlameSampler(
                n_generators=8,
                train_n_threads=train_n_threads,
                train_n_iters=train_n_iters,
                eval_n_threads=eval_n_threads,
                eval_n_iters=eval_n_iters,
            )
        ]

        self.splatters = [
            df.Splatter(
                sampler=self.samplers[0],
                train_resolution=train_resolution,
                eval_resolution=eval_resolution,
            )
        ]

        self.palettes = [df.Palette(n_qualities=self.samplers[0].n_generators)]

        self.painters = [
            df.Painter(splatter=self.splatters[0], palette=self.palettes[0])
        ]

        self.compositor = df.Compositor(painters=self.painters, bg_color=tm.vec3(1.0))

        self.train_output_buffer = self.compositor.train_output_buffer
        self.eval_output_buffer = self.compositor.eval_output_buffer

        self.mse = df.MeanSquaredError(self.train_output_buffer)

    def update(self):
        for i in range(len(self.samplers)):
            self.samplers[i].update(lr=1e-1)
            self.splatters[i].update(lr=1e-2)
            self.palettes[i].update(lr=1e2)
        self.compositor.update(lr=1e-0)


if __name__ == "__main__":
    main()
