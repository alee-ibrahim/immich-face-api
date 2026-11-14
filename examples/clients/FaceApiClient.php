<?php
// Laravel (server side) example. config/services.php:
//   'face_api' => ['url' => env('FACE_API_URL'), 'key' => env('FACE_API_KEY'), 'headers' => []],
// Call it from a queued job, not from a page request (one photo takes about 1-3 s).

namespace App\Services;

use Illuminate\Http\Client\RequestException;
use Illuminate\Support\Facades\Http;

class FaceApiClient
{
    /** @return array{faces: array<int, array>, image: array{w:int,h:int}} */
    public function identify(string $jpegBytes, int $candidates = 3): array
    {
        $response = Http::withToken(config('services.face_api.key'))
            ->withHeaders(config('services.face_api.headers', []))   // e.g. an access-proxy service token
            ->timeout(120)
            ->retry(3, 5000, fn ($e) => $e instanceof RequestException
                && in_array($e->response->status(), [429, 502, 503], true), throw: false)
            ->attach('image', $jpegBytes, 'photo.jpg')
            ->post(rtrim(config('services.face_api.url'), '/').'/v1/identify?candidates='.$candidates);

        $response->throw();   // 400/401/413/415 are bugs or bad input: log and move on

        return $response->json();
    }

    /**
     * Re-check stored tags against a photo. $personIds: Immich person ids the photo is tagged with.
     * $box: optional face box from an earlier identify. Each claim has a 'verdict':
     * match | match_not_best | no_match | no_face | no_reference | person_gone.
     */
    public function verify(string $imageBytes, array $personIds, ?array $box = null): array
    {
        $fields = [];
        foreach ($personIds as $id) {
            $fields[] = ['name' => 'person_id', 'contents' => $id];
        }
        if ($box !== null) {
            $fields[] = ['name' => 'box', 'contents' => json_encode($box)];
        }

        return $this->http()
            ->timeout(120)
            ->attach('image', $imageBytes, 'photo.jpg')
            ->asMultipart()
            ->post($this->url('/v1/verify'), $fields)
            ->throw()
            ->json();
    }

    /** Named people now: ['people' => [...], 'count' => n, 'changes_cursor' => 'pc_...' (with the change log)]. */
    public function people(): array
    {
        return $this->http()->get($this->url('/v1/people'))->throw()->json();
    }

    /** One page of people changes after $cursor (null = from the start). Save 'next_cursor'; repeat while 'has_more'. */
    public function changes(?string $cursor, int $limit = 1000): array
    {
        return $this->http()
            ->get($this->url('/v1/people/changes'), array_filter(['since' => $cursor, 'limit' => $limit]))
            ->throw()
            ->json();
    }

    /** Where stored person ids stand now: current, merged (-> current_person_id), hidden, unnamed, deleted, unknown. */
    public function resolve(array $personIds): array
    {
        return $this->http()
            ->post($this->url('/v1/people/resolve'), ['person_ids' => array_values($personIds)])
            ->throw()
            ->json('people');
    }

    private function http(): \Illuminate\Http\Client\PendingRequest
    {
        return Http::withToken(config('services.face_api.key'))
            ->withHeaders(config('services.face_api.headers', []))
            ->timeout(60)
            ->retry(3, 5000, fn ($e) => $e instanceof RequestException
                && in_array($e->response->status(), [429, 502, 503], true), throw: false);
    }

    private function url(string $path): string
    {
        return rtrim(config('services.face_api.url'), '/').$path;
    }
}

// Usage:
//   $result = app(FaceApiClient::class)->identify(Storage::get($photo->large_path));
//   foreach ($result['faces'] as $face) {
//       // $face['box'] (fractions), $face['person_id'], $face['name'] (null = unknown),
//       // $face['ambiguous'], $face['candidates']  -> store as pending suggestions for staff to confirm
//   }
//
// Nightly sync (see README "Keeping tags right"):
//   $cursor = Setting::get('face_api_cursor');
//   do {
//       $page = $client->changes($cursor);
//       if ($page['resync_required']) { /* full re-sync: GET /v1/people + resolve(), then use its changes_cursor */ }
//       foreach ($page['changes'] as $e) { /* renamed / merged / faces_moved / deleted / created */ }
//       $cursor = $page['next_cursor'];
//       Setting::put('face_api_cursor', $cursor);
//   } while ($page['has_more']);
