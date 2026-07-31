import {describe,it,expect} from 'vitest';
import {backendSplit} from '../src/lib/api';
import {serializeFeedback,type FeedbackPacket} from '../src/lib/feedback';
import {togglePane} from '../src/lib/paneState';
describe('API contracts and packet safety',()=>{it('maps product test label to backend holdout',()=>expect(backendSplit('test')).toBe('holdout'));it('keeps unmatched sides null and escapes fences',()=>{const p:FeedbackPacket={pdfPath:'/original/a.md\nnext',split:'test',page:null,bbox:null,candidate:{id:'c',json:{x:'```'},rendered:'ok'},reference:null,alignment:'unmatched candidate',comment:'line\n```'};const out=serializeFeedback(p);expect(out).toContain('BBox: null');expect(out).toContain('Element: null (unmatched / unavailable)');expect(out).toContain('``\\`');});it('supports maximize as persisted pane mode shape',()=>expect(togglePane({source:true},'source').source).toBe(false))});
