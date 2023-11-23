import os
import sys
import glob
import socket
from socket import error as SocketError
import errno
import logging
import itertools
import subprocess
import argparse

from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("hyperpixel-server")
script_dir = os.path.dirname(os.path.realpath(__file__))


class FileModifiedHandler(FileSystemEventHandler):
    def __init__(self, callback):
        self.callback = callback

    def on_modified(self, event):
        logger.info("File modified!")
        self.callback()


class HyperPixel:
    image_files = []
    current_image = None
    current_image_path = None
    feh_pid = None
    observer = None

    def __init__(self, image_path_glob):
        self.image_path_glob = image_path_glob

    def toggle_lcd_state(self, on: bool):
        script = "lcdon.sh" if on else "lcdoff.sh"
        subprocess.run(['sudo', '/bin/bash', f'{script_dir}/{script}'], check=True)

    def client_connected(self, conn, client_addr):
        logger.info('Client %s connected', ':'.join(map(str, client_addr)))
        self.image_files = list(sorted(glob.glob(self.image_path_glob)))
        logger.info('%d image files available', len(self.image_files))
        self.toggle_lcd_state(on=True)

    def kill_image(self):
        if self.feh_pid:
            # running .kill() on the Popen class messes up the calling shell process, annoying, this works better
            subprocess.run(['kill', str(self.feh_pid.pid)])
        self.feh_pid = None
        self.current_image_path = None
        if self.observer: self.observer.stop()

    def client_disconnected(self, conn, client_addr):
        self.toggle_lcd_state(on=False)
        if self.observer: self.observer.stop()
        if client_addr: logger.info('Client %s disconnected', ':'.join(map(str, client_addr)))
        if conn:
            try: conn.close()
            except: pass

    def refresh_image(self):
        if self.current_image_path and self.feh_pid:
            self.display_image(self.current_image_path)

    def display_image(self, image_path):
        logger.info('Displaying image %s', image_path)
        self.kill_image()
        self.feh_pid = subprocess.Popen(['/usr/bin/feh', '--hide-pointer', '-x', '-q', '-B', 'white', '-F', image_path],
                                        env={'DISPLAY': ':0', 'HOME': '/home/pi'})
        self.current_image_path = image_path
        self.observer = PollingObserver()
        self.observer.schedule(FileModifiedHandler(self.refresh_image), image_path, recursive=False)
        self.observer.start()

    def change_image(self, forward: bool):
        if not self.image_files: return
        n = len(self.image_files)
        cur = self.current_image
        i = (0 if cur is None else cur + 1 if forward else cur - 1) % n
        self.display_image(self.image_files[i])
        self.current_image = i

    def command_callback(self, command: str) -> str:
        if command == 'on':
            self.toggle_lcd_state(on=True)
            return 'ok'
        elif command == 'off':
            self.toggle_lcd_state(on=False)
            return 'ok'
        elif command == 'previous':
            self.change_image(forward=False)
            return 'ok'
        elif command == 'next':
            self.change_image(forward=True)
            return 'ok'
        elif command == 'ping':
            return 'pong'
        else:
            logger.error('WTF: %s', command)
            return 'WTF?'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', dest='port', type=int, help="TCP Port to bind the server to")
    parser.add_argument('--images', dest='images', help="Glob path (quoted) to image files")
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_address = (socket.gethostname() + '.local', args.port)
    sock.bind(server_address)
    try:
        sock.listen(1)
        sock.settimeout(0.5)
        hp = HyperPixel(args.images)
        hp.toggle_lcd_state(on=False)
        logger.info('Waiting for connections on %s', ':'.join(map(str, server_address)))
        while True:
            connection = None
            client_address = None
            try:
                connection, client_address = sock.accept()
                hp.client_connected(connection, client_address)
                while True:
                    data = connection.recv(1024)
                    msg = data.decode("utf-8")
                    logger.debug('received: %s', msg)
                    if data:
                        response = hp.command_callback(msg)
                        try:
                            connection.send(response.encode("utf-8"))
                        except:
                            pass
                    else:
                        break
            except socket.timeout:
                continue
            except KeyboardInterrupt:
                hp.client_disconnected(connection, client_address)
                break
            except SocketError as e:
                if e.errno != errno.ECONNRESET:
                    raise
                pass
            except Exception as e:
                logger.error('Error: %s', e)
                hp.client_disconnected(connection, client_address)
                break
    except KeyboardInterrupt:
        hp.kill_image()
    finally:
        sock.close()

    logging.info("Server shut down")

