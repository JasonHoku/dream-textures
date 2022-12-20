from multiprocessing import Queue, Process, Lock
import multiprocessing.synchronize
import enum
import traceback
import threading
from typing import Type, TypeVar, Callable, Any, MutableSet, Generator
# from concurrent.futures import Future
import site

class Future:
    """
    Object that represents a value that has not completed processing, but will in the future.

    Add callbacks to be notified when values become available, or use `.result()` and `.exception()` to wait for the value.
    """
    _response_callbacks: MutableSet[Callable[['Future', Any], None]] = set()
    _exception_callbacks: MutableSet[Callable[['Future', BaseException], None]] = set()
    _done_callbacks: MutableSet[Callable[['Future'], None]] = set()
    _responses: list = []
    _exception: BaseException | None = None
    done: bool = False
    cancelled: bool = False

    def __init__(self):
        self._response_callbacks = set()
        self._exception_callbacks = set()
        self._done_callbacks = set()
        self._responses = []
        self._exception = None
        self.done = False
        self.cancelled = False

    def result(self):
        """
        Get the result value (blocking).
        """
        def _response():
            match len(self._responses):
                case 0:
                    return None
                case 1:
                    return self._responses[0]
                case _:
                    return self._responses
        if self._exception is not None:
            raise self._exception
        if self.done:
            return _response()
        else:
            event = threading.Event()
            def _done(_):
                event.set()
            self.add_done_callback(_done)
            event.wait()
            if self._exception is not None:
                raise self._exception
            return _response()
    
    def exception(self):
        if self.done:
            return self._exception
        else:
            event = threading.Event()
            def _done(_):
                event.set()
            self.add_done_callback(_done)
            event.wait()
            return self._exception
    
    def cancel(self):
        self._cancelled = True
        self.set_done()

    def add_response(self, response):
        """
        Add a response value and notify all consumers.
        """
        self._responses.append(response)
        for response_callback in self._response_callbacks:
            response_callback(self, response)

    def set_exception(self, exception: BaseException):
        """
        Set the exception.
        """
        self._exception = exception

    def set_done(self):
        """
        Mark the future as done.
        """
        assert not self.done
        self.done = True
        for done_callback in self._done_callbacks:
            done_callback(self)

    def add_response_callback(self, callback: Callable[['Future', Any], None]):
        """
        Add a callback to run whenever a response is received.
        Will be called multiple times by generator functions.
        """
        self._response_callbacks.add(callback)
    
    def add_exception_callback(self, callback: Callable[['Future', BaseException], None]):
        """
        Add a callback to run when the future errors.
        Will only be called once at the first exception.
        """
        self._exception_callbacks.add(callback)

    def add_done_callback(self, callback: Callable[['Future'], None]):
        """
        Add a callback to run when the future is marked as done.
        Will only be called once.
        """
        self._done_callbacks.add(callback)

class ActorContext(enum.IntEnum):
    """
    The context of an `Actor` object.
    
    One `Actor` instance is the `FRONTEND`, while the other instance is the backend, which runs in a separate process.
    The `FRONTEND` sends messages to the `BACKEND`, which does work and returns a result.
    """
    FRONTEND = 0
    BACKEND = 1

class Message:
    """
    Represents a function signature with a method name, positonal arguments, and keyword arguments.

    Note: All arguments must be picklable.
    """

    def __init__(self, method_name, args, kwargs):
        self.method_name = method_name
        self.args = args
        self.kwargs = kwargs
    
    CANCEL = "__cancel__"
    END = "__end__"

def _start_backend(cls, message_queue, response_queue):
    cls(
        ActorContext.BACKEND,
        message_queue=message_queue,
        response_queue=response_queue
    ).start()

class TracedError(BaseException):
    def __init__(self, base: BaseException, trace: str):
        self.base = base
        self.trace = trace

T = TypeVar('T', bound='Actor')

class Actor:
    """
    Base class for specialized actors.
    
    Uses queues to send actions to a background process and receive a response.
    Calls to any method declared by the frontend are automatically dispatched to the backend.

    All function arguments must be picklable.
    """

    _message_queue: Queue
    _response_queue: Queue
    _lock: multiprocessing.synchronize.Lock

    _shared_instance = None

    # Methods that are not used for message passing, and should not be overridden in `_setup`.
    _protected_methods = {
        "start",
        "close",
        "is_alive",
        "can_use",
        "shared"
    }

    def __init__(self, context: ActorContext, message_queue: Queue = Queue(maxsize=1), response_queue: Queue = Queue(maxsize=1)):
        self.context = context
        self._message_queue = message_queue
        self._response_queue = response_queue
        self._setup()
        self.__class__._shared_instance = self
    
    def _setup(self):
        """
        Setup the Actor after initialization.
        """
        match self.context:
            case ActorContext.FRONTEND:
                self._lock = Lock()
                for name in filter(lambda name: callable(getattr(self, name)) and not name.startswith("_") and name not in self._protected_methods, dir(self)):
                    setattr(self, name, self._send(name))
            case ActorContext.BACKEND:
                pass

    @classmethod
    def shared(cls: Type[T]) -> T:
        return cls._shared_instance or cls(ActorContext.FRONTEND).start()

    def start(self: T) -> T:
        """
        Start the actor process.
        """
        match self.context:
            case ActorContext.FRONTEND:
                self.process = Process(target=_start_backend, args=(self.__class__, self._message_queue, self._response_queue), name="__actor__", daemon=True)
                self.process.start()
            case ActorContext.BACKEND:
                self._backend_loop()
        return self
    
    def close(self):
        """
        Stop the actor process.
        """
        match self.context:
            case ActorContext.FRONTEND:
                self.process.terminate()
                self._message_queue.close()
                self._response_queue.close()
            case ActorContext.BACKEND:
                pass
    
    @classmethod
    def shared_close(cls: Type[T]):
        if cls._shared_instance is None:
            return
        cls._shared_instance.close()
        cls._shared_instance = None
    
    def is_alive(self):
        match self.context:
            case ActorContext.FRONTEND:
                return self.process.is_alive()
            case ActorContext.BACKEND:
                return True

    def can_use(self):
        if result := self._lock.acquire(block=False):
            self._lock.release()
        return result
    
    def _load_dependencies(self):
        from ..absolute_path import absolute_path
        site.addsitedir(absolute_path(".python_dependencies"))

    def _backend_loop(self):
        self._load_dependencies()
        while True:
            self._receive(self._message_queue.get())

    def _receive(self, message: Message):
        try:
            response = getattr(self, message.method_name)(*message.args, **message.kwargs)
            if isinstance(response, Generator):
                for res in iter(response):
                    extra_message = None
                    try:
                        self._message_queue.get(block=False)
                    except:
                        pass
                    if extra_message == Message.CANCEL:
                        break
                    self._response_queue.put(res)
            else:
                self._response_queue.put(response)
        except Exception as e:
            trace = traceback.format_exc()
            self._response_queue.put(TracedError(e, trace))
        self._response_queue.put(Message.END)

    def _send(self, name):
        def _send(*args, **kwargs):
            future = Future()
            def _send_thread(future: Future):
                self._lock.acquire()
                self._message_queue.put(Message(name, args, kwargs))

                while not future.done:
                    if future.cancelled:
                        self._message_queue.put(Message.CANCEL)
                    response = self._response_queue.get()
                    if response == Message.END:
                        future.set_done()
                    elif isinstance(response, TracedError):
                        response.base.__cause__ = Exception(response.trace)
                        future.set_exception(response.base)
                    elif isinstance(response, Exception):
                        future.set_exception(response)
                    else:
                        future.add_response(response)
                
                self._lock.release()
            thread = threading.Thread(target=_send_thread, args=(future,), daemon=True)
            thread.start()
            return future
        return _send
    
    def __del__(self):
        self.close()