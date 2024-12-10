import subprocess
import os
import pygame
import time

# Define two temporary output file names
#output_file_1 = "temp_output_1.wav"
#output_file_2 = "temp_output_2.wav"


def is_file_in_use(file_path):
    """Check if a file is currently being used by another process."""
    try:
        # Try to rename the file; if it succeeds, the file is not in use
        os.rename(file_path, file_path)
        return False
    except OSError:
        return True

def safe_remove(file_path):
    """Attempt to delete a file, waiting if it's in use."""
    # Retry deletion if the file is in use
    attempts = 5
    while attempts > 0:
        if os.path.exists(file_path):
            if not is_file_in_use(file_path):
                os.remove(file_path)
                print(f"Deleted: {file_path}")
                break
            else:
                print(f"File {file_path} is in use, retrying...")
                time.sleep(0.1)  # Wait before retrying
        attempts -= 1
    else:
        print(f"Could not delete {file_path} after multiple attempts.")

async def output_with_piper(text, output_file_1, output_file_2, output_file):
    pygame.mixer.init()

    try:
        # Check if the mixer is playing and stop it
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()  # Stop playback if it's ongoing

        # Add a slight delay to ensure playback has stopped
        time.sleep(0.1)  # Adjust the duration as necessary

        # Determine the other file to delete
        other_file = output_file_1 if output_file == output_file_2 else output_file_2
        
        # Safely remove the other output file if it exists
        safe_remove(other_file)

        # Command to run the Piper executable with the recognized text and save to the current output file
        command = f'echo "{text}" | .\\piper.exe -m en_GB-cori-high.onnx -f {output_file}'

        # Run the command and wait for it to complete
        subprocess.run(command, shell=True, check=True)

        # Load and play the output audio file
        pygame.mixer.music.load(output_file)
        pygame.mixer.music.play()

        print(f"Speaking: {text}")

        # Wait until the music finishes playing
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)  # Wait for music to finish playing

    except subprocess.CalledProcessError as e:
        print(f"An error occurred while generating audio: {e}")
    except Exception as e:
        print(f"An error occurred during playback: {e}")

