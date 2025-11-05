import os, re, cv2
from typing import Union
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


class SimpleInference:
    """
    A class for performing inference using Hugging Face models.
    """

    def __init__(self, model_id="BAAI/RoboBrain2.0-7B"):
        """
        Initialize the model and processor.
        
        Args:
            model_id (str): Path or Hugging Face model identifier (default: "BAAI/RoboBrain2.0-7B")
        """
        print("Loading Checkpoint ...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, torch_dtype="auto", device_map="auto")

        self.processor = AutoProcessor.from_pretrained(model_id)

    def inference(self,
                  text: str,
                  image: Union[list, str],
                  task="general",
                  plot=False,
                  enable_thinking=True,
                  do_sample=True,
                  temperature=0.7):
        """Perform inference with text and images input.
        Args:
            text (str): The input text prompt.
            image (Union[list,str]): The input image(s) as a list of file paths or a single file path.
            task (str): The task type, e.g., "general", "pointing", "affordance", "trajectory". If "pointing", "affordance", or "trajectory" is specified, the function will automatically adjust the text prompt.
            enable_thinking (bool): Whether to enable thinking mode.
            do_sample (bool): Whether to use sampling during generation.
            temperature (float): Temperature for sampling.
        """

        if isinstance(image, str):
            image = [image]

        assert task in [
            "general", "pointing", "affordance", "trajectory", "grounding"
        ], f"Invalid task type: {task}. Supported tasks are 'general', 'pointing', 'affordance', 'trajectory', 'grounding'."
        assert task == "general" or (
            task in ["pointing", "affordance", "trajectory", "grounding"] and
            len(image) == 1), "Pointing, affordance, grounding, and trajectory tasks require exactly one image."

        if task == "pointing":
            print("Pointing task detected. We automatically add a pointing prompt for inference.")
            text = f"{text}. Your answer should be formatted as a list of tuples, i.e. [(x1, y1), (x2, y2), ...], where each tuple contains the x and y coordinates of a point satisfying the conditions above. The coordinates should indicate the normalized pixel locations of the points in the image."
        elif task == "affordance":
            print("Affordance task detected. We automatically add an affordance prompt for inference.")
            text = f"You are a robot using the joint control. The task is \"{text}\". Please predict a possible affordance area of the end effector."
        elif task == "trajectory":
            print("Trajectory task detected. We automatically add a trajectory prompt for inference.")
            text = f"You are a robot using the joint control. The task is \"{text}\". Please predict up to 10 key trajectory points to complete the task. Your answer should be formatted as a list of tuples, i.e. [[x1, y1], [x2, y2], ...], where each tuple contains the x and y coordinates of a point."
        elif task == "grounding":
            print("Grounding task detected. We automatically add a grounding prompt for inference.")
            text = f"Please provide the bounding box coordinate of the region this sentence describes: {text}."

        print(F"##### INPUT #####\n{text}\n###############")

        messages = [
            {
                "role":
                    "user",
                "content": [
                    *[{
                        "type": "image",
                        "image": path if path.startswith("http") else f"file://{path}"
                    } for path in image],
                    {
                        "type": "text",
                        "text": f"{text}"
                    },
                ],
            },
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        if enable_thinking:
            print("Thinking enabled.")
            text = f"{text}<think>"
        else:
            print("Thinking disabled.")
            text = f"{text}<think></think><answer>"

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference
        print("Running inference ...")
        generated_ids = self.model.generate(**inputs, max_new_tokens=768, do_sample=do_sample, temperature=temperature)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        if enable_thinking:
            thinking_text = output_text[0].split("</think>")[0].replace("<think>", "").strip()
            answer_text = output_text[0].split("</think>")[1].replace("<answer>", "").replace("</answer>", "").strip()
        else:
            thinking_text = ""
            answer_text = output_text[0].replace("<answer>", "").replace("</answer>", "").strip()

        if plot and task in ["pointing", "affordance", "trajectory", "grounding"]:
            print("Plotting enabled. Drawing results on the image ...")
            # extract points, boxes, or trajectories based on the task

            plot_points, plot_boxes, plot_trajectories = None, None, None

            if task == "trajectory":
                # Extract trajectory points
                trajectory_pattern = r'(\d+),\s*(\d+)'
                trajectory_points = re.findall(trajectory_pattern, answer_text)
                plot_trajectories = [[(int(x), int(y)) for x, y in trajectory_points]]
                print(f"Extracted trajectory points: {plot_trajectories}")
                image_name_to_save = os.path.basename(image[0]).replace(".", "_with_trajectory_annotated.")
            elif task == "pointing":
                # Extract points
                point_pattern = r'\(\s*(\d+)\s*,\s*(\d+)\s*\)'
                points = re.findall(point_pattern, answer_text)
                plot_points = [(int(x), int(y)) for x, y in points]
                print(f"Extracted points: {plot_points}")
                image_name_to_save = os.path.basename(image[0]).replace(".", "_with_pointing_annotated.")
            elif task == "affordance":
                # Extract bounding boxes
                box_pattern = r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
                boxes = re.findall(box_pattern, answer_text)
                plot_boxes = [[int(x1), int(y1), int(x2), int(y2)] for x1, y1, x2, y2 in boxes]
                print(f"Extracted bounding boxes: {plot_boxes}")
                image_name_to_save = os.path.basename(image[0]).replace(".", "_with_affordance_annotated.")
            elif task == "grounding":
                # Extract bounding boxes
                box_pattern = r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
                boxes = re.findall(box_pattern, answer_text)
                plot_boxes = [[int(x1), int(y1), int(x2), int(y2)] for x1, y1, x2, y2 in boxes]
                print(f"Extracted bounding boxes: {plot_boxes}")
                image_name_to_save = os.path.basename(image[0]).replace(".", "_with_grounding_annotated.")

            os.makedirs("result", exist_ok=True)
            image_path_to_save = os.path.join("result", image_name_to_save)

            self.draw_on_image(
                image[0],
                points=plot_points,
                boxes=plot_boxes,
                trajectories=plot_trajectories,
                output_path=image_path_to_save)

        return {"thinking": thinking_text, "answer": answer_text}

    def draw_on_image(self, image_path, points=None, boxes=None, trajectories=None, output_path=None):
        """
        Draw points, bounding boxes, and trajectories on an image
        
        Parameters:
            image_path: Path to the input image
            points: List of points in format [(x1, y1), (x2, y2), ...]
            boxes: List of boxes in format [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
            trajectories: List of trajectories in format [[(x1, y1), (x2, y2), ...], [...]]
            output_path: Path to save the output image. Default adds "_annotated" suffix to input path
        """
        try:
            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"Unable to read image: {image_path}")

            # Draw points
            if points:
                for point in points:
                    x, y = point
                    cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  # Red solid circle

            # Draw bounding boxes
            if boxes:
                for box in boxes:
                    x1, y1, x2, y2 = box
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box, line width 2

            # Draw trajectories
            if trajectories:
                for trajectory in trajectories:
                    if len(trajectory) < 2:
                        continue  # Need at least 2 points to form a trajectory
                    # Connect trajectory points with lines
                    for i in range(1, len(trajectory)):
                        cv2.line(image, trajectory[i - 1], trajectory[i], (255, 0, 0), 2)  # Blue line, width 2
                    # Draw a larger point at the trajectory end
                    end_x, end_y = trajectory[-1]
                    cv2.circle(image, (end_x, end_y), 7, (255, 0, 0), -1)  # Blue solid circle, slightly larger

            # Determine output path
            if not output_path:
                name, ext = os.path.splitext(image_path)
                output_path = f"{name}_annotated{ext}"

            # Save the result
            cv2.imwrite(output_path, image)
            print(f"Annotated image saved to: {output_path}")
            return output_path

        except Exception as e:
            print(f"Error processing image: {e}")
            return None


if __name__ == "__main__":

    model = SimpleInference("BAAI/RoboBrain2.0-7B")

    prompt = "What is shown in this image?"
    image = "http://images.cocodataset.org/val2017/000000039769.jpg"

    pred = model.inference(prompt, image, task="general", plot=False, enable_thinking=True, do_sample=True)
    print(f"Prediction:\n{pred}")
