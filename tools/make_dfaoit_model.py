"""Shape-specializes the DFAOIT network of `DEMO_004_NeuralOIT` for a render resolution.

`tools/train_dfaoit.py` writes `third-party/content/src/dfaoit/dfaoit.vgf` unspecialized: a trained network says nothing about
the resolution it will run at, so no tensor of the graph carries a shape, the way Arm publishes its own networks. That
model cannot be run as it stands, because `VK_ARM_data_graph` refuses a graph whose tensor types have no shape
(VUID-RuntimeSpirv-pNext-09919). This script gives it one:

    python tools/make_dfaoit_model.py --render 1920x810

writes `third-party/content/src/dfaoit/dfaoit-1920x810.vgf`, which the sample loads. The height is not the height of the
window: the sample only puts the pixels that have more fragments than the network keeps exactly into the tensor, so the
shape it needs is `kRenderWidth x kTensorHeight` of `samples/DEMO_004_NeuralOIT.cpp`.

The shapes are trivial to work out. Every operator of the network is a 1x1 CONV2D with unit stride and no padding, or a
RESCALE or a TABLE, so all of them keep the height and the width of the input and only the channel count changes, from
the weights of each CONV2D. The graph module is disassembled, every operator result and the interface get a shaped
tensor type, and `vgf_respecialize` rebuilds the VGF around the new module. Nothing else changes: not the weights, not
the operator attributes, not the element types. The heavy lifting is shared with `make_nss_model.py`.

Needs the Vulkan SDK (`spirv-dis`, `spirv-as`, `spirv-val`) and the VGF library from `deploy_deps.py`.
"""
import os
import re
import sys

from make_nss_model import (Module, buildTool, dumpResources, extractModule, fail, findTool, option, run, specialize)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTENT = os.path.join(ROOT, 'third-party', 'content', 'src', 'dfaoit')

CONV2D_WEIGHT = 6


def inferMlp(module, inputShape):
    """Every operator keeps the height and the width; CONV2D sets the channel count from its weights [OC, 1, 1, IC]."""
    inputId, _ = module.graphInput()
    shapes = {inputId: list(inputShape)}
    current = list(inputShape)

    for resultId, _, opcode, operands, _ in module.ops():
        if opcode == 'CONV2D':
            weights = module.shapeOfValue(operands[CONV2D_WEIGHT])
            if not weights or len(weights) != 4:
                fail(f'the weights of {resultId} have no shape')
            current = [current[0], current[1], current[2], weights[0]]
        elif opcode not in ('RESCALE', 'TABLE'):
            fail(f'unexpected operator `{opcode}`: this network is meant to be 1x1 convolutions, rescales and a table')
        shapes[resultId] = list(current)

    return shapes


def numInputChannels(module):
    """The input channel count of the network, taken from the weights of its first CONV2D."""
    for _, _, opcode, operands, _ in module.ops():
        if opcode == 'CONV2D':
            weights = module.shapeOfValue(operands[CONV2D_WEIGHT])
            if weights and len(weights) == 4:
                return weights[3]
    fail('the module has no CONV2D to take the input channel count from')


def main():
    render = option('--render', '1920x1080').lower()
    if not re.fullmatch(r'\d+x\d+', render):
        fail('--render takes <width>x<height>, for example 1920x1080')
    width, height = (int(v) for v in render.split('x'))

    model = option('--source', os.path.join(CONTENT, 'dfaoit-16x16.vgf'))
    buildDir = option('--build-dir', os.path.join(ROOT, 'build'))
    stem = os.path.splitext(os.path.basename(model))[0]
    output = option('--output', os.path.join(CONTENT, f'{stem}-{width}x{height}.vgf'))

    if not os.path.exists(model):
        fail(f'`{model}` is missing. Run `tools/train_dfaoit.py` first.')
    if os.path.abspath(output) == os.path.abspath(model):
        fail('the output would overwrite the source model: pick another --render or --output')

    spirvAs, spirvDis, spirvVal = findTool('spirv-as'), findTool('spirv-dis'), findTool('spirv-val')
    tool = buildTool(buildDir)

    work = os.path.join(buildDir, 'dfaoit-model')
    os.makedirs(work, exist_ok=True)
    extracted = os.path.join(work, 'dfaoit_source.spv')
    open(extracted, 'wb').write(extractModule(model))
    module = Module(run([spirvDis, '--no-color', extracted]).splitlines())

    inputShape = [1, height, width, numInputChannels(module)]
    lines, outputShapes = specialize(module, inputShape, inferMlp)
    print(f'inferred input {inputShape} -> {" and ".join(str(s) for s in outputShapes)}')

    resources, inputSlots, outputSlots = dumpResources(tool, model)
    shapes = {inputSlots[0]: inputShape}
    for slot, shape in zip(outputSlots, outputShapes):
        shapes[slot] = shape

    assembly = os.path.join(work, f'dfaoit_{width}x{height}.spvasm')
    spirv = os.path.splitext(assembly)[0] + '.spv'
    open(assembly, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    run([spirvAs, '--target-env', 'spv1.6', '-o', spirv, assembly])
    run([spirvVal, '--target-env', 'vulkan1.3', spirv])
    print(f'{os.path.getsize(spirv)} bytes of SPIR-V, validated')

    args = [f'{index}:{",".join(str(d) for d in shape)}' for index, shape in sorted(shapes.items())]
    print(run([tool, model, spirv, output, *args]).strip())
    print(output)


if __name__ == '__main__':
    main()
