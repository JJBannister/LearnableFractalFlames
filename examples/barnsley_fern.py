import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu)

import dfractal as df


def main():
    pipe = FernPipeline(
        train_n_threads=1000,
        train_n_iters=1000,
        eval_n_threads=100000,
        eval_n_iters=100000,
        train_resolution=(200, 200),
        eval_resolution=(1200, 1200),
        greyscale_mode=True,
    )

    # Get true image and set the reference image
    pipe.initialize_parameters()
    pipe.forward()
    true_image = pipe.train_output_buffer.to_numpy()
    pipe.mse.set_reference_image(true_image)

    # Perturb the parameters
    pipe.perturb_parameters()
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
    df.plot_image(initial_image, title="initial image")
    df.plot_image(final_image, title="final image")
    df.show_plots()

    # Generate Eval Image
    print("Generating Full Resolution Image")
    pipe.forward_eval()
    eval_image = pipe.eval_output_buffer.to_numpy()
    df.plot_image(eval_image, title="eval image")
    df.show_plots()


class FernPipeline(df.Pipeline):
    def __init__(
        self,
        train_n_threads: int,
        train_n_iters: int,
        eval_n_threads: int,
        eval_n_iters: int,
        train_resolution: int,
        eval_resolution: int,
        greyscale_mode=True,
    ):

        self.train_resolution = train_resolution
        self.eval_resolution = eval_resolution

        self.samplers = [
            FernSampler(
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

        self.palettes = [
            df.Palette(
                n_qualities=self.samplers[0].n_generators, greyscale_mode=greyscale_mode
            )
        ]

        self.painters = [
            df.Painter(splatter=self.splatters[0], palette=self.palettes[0])
        ]

        self.compositor = df.Compositor(painters=self.painters, bg_color=tm.vec3(0.0))

        self.train_output_buffer = self.compositor.train_output_buffer
        self.eval_output_buffer = self.compositor.eval_output_buffer

        self.mse = df.MeanSquaredError(self.train_output_buffer)

    def update(self):
        self.samplers[0].update(lr=1e-4)

    def initialize_parameters(self):
        self.samplers[0].initialize_parameters()
        self.splatters[0].matrix[None] = tm.mat2(
            [
                [0.15, 0],
                [0, 0.15],
            ]
        )
        self.splatters[0].translation[None] = tm.vec2([0.0, -0.8])

    def perturb_parameters(self):
        self.samplers[0].perturb_parameters()


class FernSampler(df.LinearFlameSampler):

    def __init__(
        self,
        train_n_threads: int,
        train_n_iters: int,
        eval_n_threads: int,
        eval_n_iters: int,
    ):

        # Hard coded variables for the fern fractal
        n_generators = 4
        generator_weights = [1.0, 85.0, 7.0, 7.0]

        super().__init__(
            n_generators=n_generators,
            train_n_threads=train_n_threads,
            train_n_iters=train_n_iters,
            eval_n_threads=eval_n_threads,
            eval_n_iters=eval_n_iters,
            generator_weights=generator_weights,
        )

    @ti.func
    def compute_sample_position(self, generator_index: int, old_sample: tm.vec2):
        return self.apply_linear_transform(generator_index, old_sample)

    def initialize_parameters(self):
        self.matrices[0] = tm.mat2(
            [
                [0, 0],
                [0, 0.16],
            ]
        )
        self.translations[0] = tm.vec2([0, 0])

        self.matrices[1] = tm.mat2(
            [
                [0.85, 0.04],
                [-0.04, 0.85],
            ]
        )
        self.translations[1] = tm.vec2([0, 1.6])

        self.matrices[2] = tm.mat2(
            [
                [0.2, -0.26],
                [0.23, 0.22],
            ]
        )
        self.translations[2] = tm.vec2([0, 1.6])

        self.matrices[3] = tm.mat2(
            [
                [-0.15, 0.28],
                [0.26, 0.24],
            ]
        )
        self.translations[3] = tm.vec2([0, 0.44])

    def perturb_parameters(self):
        self.matrices[1] = tm.mat2(
            [
                [0.85, -0.00],
                [0.00, 0.85],
            ]
        )


if __name__ == "__main__":
    main()
