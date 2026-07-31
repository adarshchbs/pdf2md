export type PaneMode={source?:boolean;candidate?:boolean;reference?:boolean};
export function readPaneState(storage:Storage,key='pdf2md-pane-state'):PaneMode{try{return JSON.parse(storage.getItem(key)||'{}') as PaneMode}catch{return {}}}
export function writePaneState(storage:Storage,state:PaneMode,key='pdf2md-pane-state'){storage.setItem(key,JSON.stringify(state));return state}
export function togglePane(state:PaneMode,name:keyof PaneMode):PaneMode{return {...state,[name]:state[name] !== true}}
