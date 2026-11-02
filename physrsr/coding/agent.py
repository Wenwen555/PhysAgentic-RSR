"""Codex CLI backend, isolated from repository and evaluator files."""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

SCHEMA = {'type':'object','properties':{'code':{'type':'string'},'rationale':{'type':'string'}},
          'required':['code','rationale'],'additionalProperties':False}


def system_mounts():
    args=[]
    for name in ('/usr','/bin','/lib','/lib64'):
        if Path(name).exists(): args += ['--ro-bind',name,name]
    return args


@dataclass
class CodexAgent:
    model: str = 'gpt-6-astra'
    effort: str = 'low'
    timeout: int = 300

    def generate(self, messages, destination, *, schema=None):
        destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
        (destination/'messages.json').write_text(json.dumps(messages,indent=2))
        binary=Path(shutil.which('codex') or '').resolve()
        auth=Path.home()/'.codex/auth.json'
        if not binary.is_file() or not auth.is_file() or not shutil.which('bwrap'):
            raise RuntimeError('Codex CLI, login and bubblewrap are required')
        prompt=('Generate Python code only through the requested JSON schema. Do not use tools, inspect files, '
                'or contact any external service. Conversation:\n'+json.dumps(messages))
        with tempfile.TemporaryDirectory(prefix='physrsr-codex-') as temporary:
            work=Path(temporary);(work/'schema.json').write_text(json.dumps(SCHEMA if schema is None else schema))
            # Only executable, system runtime, authentication and empty work dir
            # are mounted. No repo, evaluator artifacts, user config or skills.
            command=['bwrap','--die-with-parent','--new-session','--unshare-user','--unshare-pid',
                     '--unshare-ipc','--unshare-uts','--cap-drop','ALL',*system_mounts(),
                     '--proc','/proc','--dev','/dev','--tmpfs','/tmp',
                     '--dir',str(Path.home()/'.codex'),'--ro-bind',str(auth),str(auth),
                     '--ro-bind',str(binary),'/codex','--bind',str(work),'/work','--chdir','/work']
            for name in ('/etc/resolv.conf','/etc/hosts','/etc/ssl'):
                if Path(name).exists():command += ['--ro-bind',name,name]
            command += ['/codex','exec','--disable','code_mode_host','--disable','shell_tool',
                        '--disable','apps','--disable','multi_agent','--ignore-user-config','--ignore-rules','--ephemeral',
                        '--skip-git-repo-check','--sandbox','read-only','--json','--color','never',
                        '--model',self.model,'-c',f'model_reasoning_effort="{self.effort}"',
                        '--output-schema','/work/schema.json','--output-last-message','/work/answer.json','-']
            # Preserve existing HOME for Codex login discovery; never pass project
            # env vars, model credentials, session IDs, or hidden physical data.
            environment={k:v for k,v in os.environ.items() if k in (
                'HOME','PATH','LANG','HTTPS_PROXY','HTTP_PROXY','ALL_PROXY','NO_PROXY',
                'https_proxy','http_proxy','all_proxy','no_proxy')}
            start=time.perf_counter()
            try:
                completed=subprocess.run(command,input=prompt,text=True,capture_output=True,
                                         timeout=self.timeout,env=environment)
            except subprocess.TimeoutExpired as exc:
                (destination/'error.txt').write_text('Codex inference timed out')
                raise RuntimeError('Codex inference timeout; no program substituted') from exc
            (destination/'events.jsonl').write_text(completed.stdout)
            (destination/'stderr.txt').write_text(completed.stderr)
            metadata=dict(model=self.model,reasoning_effort=self.effort,
                          codex_version=subprocess.check_output([str(binary),'--version'],text=True).strip(),
                          disabled_features=['code_mode_host','shell_tool','apps','multi_agent'],
                          duration_s=time.perf_counter()-start,returncode=completed.returncode)
            events=[]
            for line in completed.stdout.splitlines():
                try:events.append(json.loads(line))
                except json.JSONDecodeError:pass
            metadata['usage']=[e.get('usage') for e in events if e.get('type')=='turn.completed']
            (destination/'metadata.json').write_text(json.dumps(metadata,indent=2))
            answer=work/'answer.json'
            if completed.returncode or not answer.exists():
                raise RuntimeError(f'Codex invocation failed; see {destination}/stderr.txt')
            payload=json.loads(answer.read_text())
            if not isinstance(payload,dict):
                raise RuntimeError('Invalid Codex object response')
            if schema is None and (set(payload)!={'code','rationale'} or not all(isinstance(v,str) for v in payload.values())):
                raise RuntimeError('Invalid Codex output schema')
            (destination/'answer.json').write_text(json.dumps(payload,indent=2))
            if schema is None:
                (destination/'program.py').write_text(payload['code'])
            else:
                (destination/'schema.json').write_text(json.dumps(schema,indent=2))
            return payload
